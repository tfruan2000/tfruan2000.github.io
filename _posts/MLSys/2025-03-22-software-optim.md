---
title: Software Optimization
author: tfruan
date: 2025-03-22 12:00:00 +0800
categories: [MLSys]
tags: [MLSys, Optimization]
---

# 优化原则

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

# Operation Fuse

算子融合在训练和推理中都是重要的技术，能够减少中间变量的存储。

- 训练：减少内存带宽占用。需要考虑梯度因素的影响，所以有些中间结果必须要保存。需要考虑**混合精度训练**等带来的影响。
- 推理：减少内存读写，提升计算图执行速度。

# Loop Pipelining

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

# Redundant Eliminate

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
