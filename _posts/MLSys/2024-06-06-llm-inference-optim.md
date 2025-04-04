---
title: LLM Inference Optimization
author: tfruan
date: 2024-06-06 12:00:00 +0800
categories: [MLSys]
tags: [MLSys, LLM, Optimization]
---

# LLM Basic Note

## 常见指标

- TPS: Token Per Second
- MFU: Model FLOPs Utilization
- MaaS: Model as a service
- TTFT: Time To First Token
- TPOC: Time Per Output Token

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

prefill 阶段的输入是已知的完整序列，模型需要同时看到所有 token 来提取全局信息。所以一般不需要 mask。

而 decode 阶段是自回归，需要限制模型只能看到当前 token 及其之前的 token 来生成下一个 token，因此会使用 mask (例如 casual mask)。

5.kvcache

kvcache只适用于decoder self-attention结构，因为decoder结构有casual_mask，self-attention的KV需要按步更新。

以空间换时间。

![cache](/assets/img/blog/img_llm_inference_optim/cache.png)

- 传统注意力计算时，需要将当前时刻的 `Q` 去乘以过去所有时刻的 `K`，如果没有KV Cache，每一次就需要重新计算KV(通过输入和权重矩阵计算)。
- 每一轮计算只依赖于当前的 Q_i，以及之前所有时刻的 K、V，使用kv cache后，就不需要重新计算之前的KV
- Q Cache：从单request来讲是不需要Q cache的，但从整体系统而言，分析每次 request 的 Q 相关性（例如问一样的问题），也是对系统有意义的。

GEMM 降级到 GEMV 后，还想继续用上 tensor core，可以增加计算的并行，将运算拼接成一个GEMM。

window attn(sliding windows)：每个token只和包含本身在内的前n个token做attn（使用前几次decode的kvcache），这样kvcache就可以只存一个window的个数。更早的token信息可以跨层流动，就像cnn中的感受野（输出只跟一些输入有关）

> window attn 是在 Mixtral 中提出的，Mixtral 中Rotary的意思就是：通过某种规则，将Cache中的数据旋转回正确的位置，以便能正确做Attention。因为cache每次都在更新，重新写入，所顺序是不符合期望的。

kv cache存储会造成大量碎片化 -> 使用分页管理(page):将每个序列的键值划分为块，采用非连续的存储分配方案，减少空间浪费

6.MQA、GQA、MLA

- MQA：所有 Attention-Head 共享一组 KV Cache
- GQA：一组 KV Cache 支持一个 Group 的 Q
- MLA：不再做 QK 的乘积，直接使用投影表示。无法使用RoPE位置编码，而使用ALIBI这种。

## 训练概念

1.pretrain、SFT、RLFH

pretrain 是在大量无标签文本数据上进行模型训练的过程。模型通过预测下一个token，不断调整自身参数。

SFT全称Supervised Fine-Tuning，是在 pretrain 之后，对 response 行为进行预测调整。

RLHF全称 Reinforcement Learning from Human Feedback，让模型生成的内容更符合人类偏好。

pretrain 可以理解为积累专业知识， SFT 可以理解学会执行特定任务，RLHF 是让模型做更符合人类期望的事情。

2.将一个基础模型训练成领域模型的步骤

要将基础模型改为领域特定模型，通常需要以下步骤：

(1) 领域数据收集，并进行数据预处理

(2) 使用领域数据进行 Pretraining

(3) 进行 SFT(Supervised Fine-Tuning)，让其在特定任务中表现得更好

(4) 训练 RM(Reward Modeling)，来评估生成内容的质量，去除得分低的

(5) RLHF使用强化学习结合人类反馈，调整模型以生成更符合需求的输出。

(6) 评估、验证、部署、监控

## 推理概念

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

3.在线推理和离线推理

- 在线推理
  - 需求：实时请求，高并发
  - 技术：动态批处理(合理调整batch，连续批处理)，内存优化(kv cache管理)
  - vLLM
- 离线推理
  - 需求：极致性能，单卡内存
  - 技术：量化、静态批处理(等待到一定量的请求后一起发出，等所有的请求都完成后才释放硬件资源)
  - TensorRT-LLM

4.投机采样(Speculative Decoding)

(1)用小模型做自回归连续采样生成n个token

(2)将生成的n个token和前缀拼接在一起，送入大模型执行一次前向

(3)根据结果判断是接受小模型的结果还是重新进行(1)

计算量是不变的，但大模型内n个会同时参与计算，计算访存比显著提升

5.Parallelism

- DP: 一般在batch维度（独立的样本数据）拆分，将独立的数据拆开
- TP: 将weight拆分到不同的设备，在不同设备上计算一部分activation，再通过通信原语给合并起来
- PP: 拆出mini-batch，每个设备负责一部分
  - 1F1B（最后一个设备算完1F后就算1B，减少activation所占显存）， 1F1B Interleaved
  - CPP(Circal Pipeline Parallel) 是PP的一种，通过引入“循环”机制，使得数据在流水线中循环流动。允许new batch在old batch计算完成之前进入流水线。
- SP: 切分Sequence，序列指的是具有时间或顺序关系的数据，例如文本中的单词序列、时间序列数据
  - 当sequence被分到不同devie上，需要让一定的通信来传递顺序信息（attention需要关注到前一段seq的最后一个token）
- CPP: Chunk Prefill Pipeline

## 分布式

- 数据并行：数据维度（batch 维度）

- 序列并行：序列维度（sequence 维度）
  - 序列之间有前后关系，理论上应该串行计算，若要分布式：
    - 将序列划分为多个子序列时，每个子序列之间保留一定的重叠部分。
    - 设备之间通过通信机制（如 All-Gather 或 All-Reduce）交换边界 token 的信息，以计算跨设备的注意力分数。

## 量化

量化：低位宽数据代替高位宽数据，激活值很难量化（存在异常值，导致量化后精度损失严重，所以量化系统不能全模型统一，按照量化难易程度进行分块）

常见的两种形式：block-wise 和 tile-wise

- block-wise
  - 将数据分成大块，所有块共享量化参数
  - 存储开销小，计算复杂度低
- tile-wise
  - 将数据为为多个小块，每个小块使用单独的量化参数
  - 对于数据分布不均匀的场景能更好地防止精度下降

# Software Optim

## 优化原则

io瓶颈还是计算瓶颈，根据 arch 的访存计算比，随算子规模变化而改变。算子的计算量是恒定的，io可能冗余

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

软件流水一般将指令分为 prologue、computation、epilogue 三类。流水展开针对的是 for 循环。

过程：

1.构建依赖关系图

使用 bool 类型的二维数组记录 op 之间的关系，这里不记录标量计算，如果在 memref 上，则需要考虑 alias 关系。

2.划分stage

同 stage 内的 op 要求:

(1) 无依赖

(2) 有依赖但占用相同的计算资源(如都占用向量计算资源 / io资源)

当相邻 op 存在数据依赖且占据不同计算资源时，如 copy + add ，那么将其分到不同的 stage 上去

3.分析需要跨 stage 的值及其份数

通过多消耗资源来解除数据依赖，要分析空间复用，不能无脑 alloc

4.分析stage 之间的 delay

根据 stage 之间的依赖关系分析 delay，默认情况下 delay = 0，例如对 scf.forOp 的 iter_args 参数，进行修改和使用

5.展开(重写 for 循环)

- Iteration的数量小于stage数量，因此无法形成Kernel部分，会进行完全展开，op会被复制Iteration次

- 带有kernel的静态展开

```text
scf.for %arg0 = %c0 to %c4 step %c1 {
    Stage 0
    Stage 1
    Stage 2
}

->

scf.if %true {
    Stage 0

    sync
    Stage 1
    Stage 0

    sync
    scf.for %arg0 = %c0 to %c2 step %c1 {
        Stage 2
        Stage 1
        Stage 0
        sync
    }
    Stage 2
    Stage 1

    sync
    Stage 2

    sync
}
```

## Redundant Eliminate

```text
def s
copy s -> t
use t

# 前向删除
def s
use s
t无后续use


# 后向删除
def t
t无use
use t
s无后续use
```
