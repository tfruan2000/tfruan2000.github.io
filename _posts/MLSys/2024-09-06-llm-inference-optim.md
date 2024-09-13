---
title: LLM Inference Optimization
author: tfruan
date: 2024-09-06 12:00:00 +0800
categories: [MLSys]
tags: [MLSys, LLM, Optimization]
---

# LLM Basic Note

## attention

1.QK的相似度计算不用除以模长

点积直接表现了向量之间的相似性，在后序softmax计算中，越大的点积就能越突出（越大的权重）。

如果除以模长，就变成了余弦相似度，忽略了向量长度因素的影响。

在self-attention和MHA中，Q和K都是通过变换而来的，保留模长的信息保证了在训练中可以改变变换Q和K的变换参数。

所以处理模长当前只增加了计算复杂度，并未带来显著的收益。

2.除以 $\sqrt{d}$ 得作用

缩放因子用于平衡不同维度的影响，**使得softmax输出更加平滑**，防止输出过于尖锐，导致梯度消失/爆炸。

选择 $\sqrt{d}$ 而不是 $d$ 作为缩放因子，是一种权衡，既保证了点积的值不会太大，也保证了softmax中有一定区分度。

3.softmax激活函数

softmax会将最后的结果归一化，相当于概率分布(落在0～1之间)，全部的结果加起来为1。

这反应的是每个Key对Query的重要性。

其他激活函数不能满足这两点，或者要增加额外的计算，而且softmax梯度计算简单。

4.mask

attention(decoder) 中的 mask：因为是对初始输入一个个计算，当对token_i进行计算时，需要舍弃掉其后的token，所以需要mask来做舍去的行为(softmax时会认为-inf的部分是0)，**以排除干扰**

encoder的设计初衷是给 decoder 提供完整输入序列的 feature 表示。而 decoder 会逐 token 输出，只在乎时间 i 及之前的 token 信息。

5.kvcache

kvcache只适用于decoder self-attention结构，因为decoder结构有casual_mask，self-attention的KV需要按步更新。

以空间换时间。

![cache](/assets/img/blog/img_llm_inference_optim/cache.png)

- 传统注意力计算时，需要将当前时刻的 `Q` 去乘以过去所有时刻的 `K`，如果没有KV Cache，每一次就需要重新计算KV。
- 使用kv cache后，就不需要重新计算之前的KV Cache。

GEMM 降级到 GEMV 后，还想继续用上 tensor core，可以增加计算的并行，将运算拼接成一个GEMM。

window attn：每个token只和包含本身在内的前n个token做attn（使用前几次decode的kvcache），这样kvcache就可以只存一个window的个数。更早的token信息可以跨层流动，就像cnn中的感受野（输出只跟一些输入有关）

> window attn 是在 Mixtral 中提出的，Mixtral 中Rotary的意思就是：通过某种规则，将Cache中的数据旋转回正确的位置，以便能正确做Attention。因为cache每次都在更新，重新写入，所顺序是不符合期望的。

kv cache存储会造成大量碎片化 -> 使用分页管理(page):将每个序列的键值划分为块，采用非连续的存储分配方案，减少空间浪费

6.MQA、GQA、MLA

- MQA：所有 Attention-Head 共享一组 KV Cache
- GQA：一组 KV Cache 支持一个 Group 的 Q
- MLA：不再做 QK 的乘积，直接使用投影表示。无法使用RoPE位置编码，而使用ALIBI这种。

## 推理

1.prefill 和 decode

推理过程分为prefill和decode，只不过decode是逐个生成token，不能像prefill那样大段prompt做并行。

- prefill：模型理解用户的输入，需要吃下完整一段数据
  - 两个目标：生成 Prompt token 时的 KV Cache + 生成首个 token
  - Q K V 序列长度相同，一般是 计算密集型(计算瓶颈)
  - 常使用 Flash-Attention 算子优化这方面
- decode：模型逐token生成回答(续写)
  - 在原始序列上继续(续写)生成 token
  - Q 一般远小于 K 和 V，是 访存密集型(访存瓶颈)，每次生成 token 都要去读已经存储的 KV Cache
  - 常使用 Page-Attention 的技术去优化 KV Cache 的存储。

2.decoder only

当前LLM的架构基本都是decoder only，好处是训练效率高：方便并行（特别是流水并行），加大参数量更简单

3.量化

量化：低位宽数据代替高位宽数据，激活值很难量化（存在异常值，导致量化后精度损失严重，所以量化系统不能全模型统一，按照量化难易程度进行分块）

4.在线推理和离线推理

- 在线推理
  - 需求：实时请求，高并发
  - 技术：动态批处理(合理调整batch，连续批处理)，内存优化(kv cache管理)
  - vLLM
- 离线推理
  - 需求：极致性能，单卡内存
  - 技术：量化、静态批处理
  - TensorRT-LLM

# Software Optim

## 优化原则

对 SIMD 硬件的优化 和 SIMT 硬件的优化

- SIMD
  - latency bound 优化，越快完成越好 -> 保证访存连续性，用连续指令(非strided，非scalar)
  - 其他优化：
    - tile(+fuse) 到不同 core 上并行执行，core之间利用smem交换数据 -> 减少 data move
    - 在core内循环展开(最内维)做软流水；core之间async -> 减少访存 latency
- SIMT
  - throughtput bound 优化，吞吐越大越大 -> 用好 DMA 和 TMA，打满 tensorcore
  - 其他优化：
    - 离散访存优化 smem 中 memory-coalesce（warp内的thread在同一时刻执行的指令是相同的，所以要减少指令的下发）、layout-swlizzed -> 减少访存 latency
    - 异步调度 warp，通过wrap切换来掩盖访存延迟(其实相当于软流水) -> 减少访存 latency

> core 之间 async  <--> warp 之间 async
>
> 访存优化三板斧：减少数据搬运(用上smem，都写完然后IO出去)，减少数据访存延迟(软流水+减少bank conlict)，保证负载均衡
{: .prompt-info }

## Operation Fuse

算子融合在训练和推理中都是重要的技术，能够减少中间变量的存储。

- 训练：减少内存带宽占用。需要考虑梯度因素的影响，所以有些中间结果必须要保存。需要考虑**混合精度训练**等带来的影响。
- 推理：减少内存读写，提升计算图执行速度。

## Loop Pipelining

mlir 中的任务调度和软流水一般在 hardware dialect 做，接近直接操纵汇编，所以指令排布的自由度比较大。而软件流水本就是为了掩盖访存延迟。

软件流水是从最内层展开的(最内层for循环展开)，将片外数据传递、片上访存、计算指令根据依赖排开。

软件流水一般将指令分为 prologue、computation、epilogue 三类。
