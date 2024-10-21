title: OpenAI Triton 源码走读[ttir-2-ttgir]

相比 `ttir` 专注表示计算逻辑(硬件无关)，`ttgir` 表示和硬件相关的计算表示。新增的 op 例如：

- alloc_tensor : tensor<128x32xf32, #share> ：申请shared memory
- insert_slice_async： 往 shared memory 上 insert 一个 slice，语意上类似 `tensor.insert_slice`，底层对应的cp.async指令
- async_commit_group： 底层对应的 cp.async.commit_group 指令，就是将前面的所有 cp.async 指令打包到一起执行
- async_wait {num}： 底层对应的 cp.async.wait_group 指令，也就是等待前面 num 个 cp.async-groups 执行完
- cnvert_layout： 改变 tensor 的 layout

这些 op 在后续 transforms(pipeline, prefetch等) 中将很重要。本文主要介绍 ttir 到 ttgir 的 conversion，主要源码在 `createConvertTritonToTritonGPUPass` 中，需要使用来自 kernel 的 `numWarps`, `threadsPerWarp`, `numCTAs` 参数，这些信息会随着 `TritonGPUTypeConverter` 在 conversion 的过程中被用到。

- [] layout
- [] analysis(e.g. AxisInfo)
- [] Conversion

# layout

> include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td

TritonGPU Dialect 这一层的 IR的 tensor 表示将带有 Layout 的 Attr，该 Attr 定义了 Data 是如何被 Thread 并行处理。这种layout attr在lowering过程中被传递。

自 [[RFC] To generalize TritonGPU dialect for GPU of different vendors](https://github.com/triton-lang/triton/issues/2639) 后， TritonGPU Dialect 中的 Layout Attribute 正式统一为下图：

![TritonGPU Attr](/assets/img/blog/img_triton_pass/layout_attr.png)

当前最重要的也是 distributed layout 和 shared layout。

## cta layout

描述 CGA 中的 CTA 布局。主要有 `CTAsPerCGA`、`CTASplitNum`、`CTAOrder` 三个属性。

一个 tensor 被分给 `CTAPerCGA` 个 thread blocks 处理，每个 CTA 负责 `tensor_shape / CTASplitNum` 的数据。

例如对于一个 matmul AxB=C, 操作数 A = [M, K], B = [K, N], C = [M, N]：

- A、B、C 的 `CTAsPerCGA` 相同，设为 [SplatM, SplatN]
- `CTASplitNum`：A = [SplatM, 1], B = [1, SplatN], C = [SplatM, SplatN]

由于 CGA 是在 Hopper(sm90) 架构才引入的，所以该 Attr 在大部分的情况都是直接使用默认的 `CTAsPerCGA = CTASplitNum = [1...1]`。

## shared layout

表示了一种 tensor 编码模式，用于指导如何进行 `swizzle`，使得不同 thread 在访问 tensor(on shared memory) 上的元素时尽量避免 bank conflict。

> swizzle：调整数据在 shared memory 上的存储位置，保证 thread 访问时不出现 bank conflict。

该layout中主要对象为：

- vec：同行的元素进行swizzle时，连续vec个为一组
- perPhase：一次 swizzling phase 处理的 row 数。连续的 perPhase 行用相同的swizzle方法
- maxPhase：swizzling phase的个数
- order：表明哪一个为主序，[1, 0] 为行主序，同行相邻元素地址连续
- hasLeadingOffset：默认为false，Hopper MMAv3为true

swizzle 最基础的方法就是地址 与 phase_id 进行 xor。

假设现在有4x4个元素需要存到 shared memory 上去，存放的初始地址为：

[r, c] 上对应的元素为 (r:c)。

```text
[[(0:0),(0:1),(0:2),(0:3)]
[ (1:0),(1:1),(1:2),(1:3)]
[ (2:0),(2:1),(2:2),(2:3)]
[ (3:0),(3:1),(3:2),(3:3)]]
```

- #shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>

不同行中的地址与不同的参数 xor。`out[r][c] = in[r][c ^ r]`

```text
[[(0:0),(0:1),(0:2),(0:3)]  // phase 0 (xor with 0)
[ (1:1),(1:0),(1:3),(1:2)]  // phase 1 (xor with 1)
[ (2:2),(2:3),(2:0),(2:1)]  // phase 2 (xor with 2)
[ (3:3),(3:2),(3:1),(3:0)]] // phase 3 (xor with 3)
```

- #shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>

相邻 2 (`perPhase`) 行中的地址用相同的参数xor。`out[r][c] = in[r][c ^ (r / 2)]`

```text
[[(0:0),(0:1),(0:2),(0:3)]   // phase 0 (xor with 0)
[ (1:0),(1:1),(1:2),(1:3)]   // phase 0 (xor with 0)
[ (2:1),(2:0),(2:3),(2:2)]   // phase 1 (xor with 1)
[ (3:1),(3:0),(3:3),(3:2)]]  // phase 1 (xor with 1)
```

- #shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>

每隔 2(`maxPhase`) 行，地址用相同的参数xor。`out[r][c] = in[r][c ^ (r % 2)]`

> `phase` 相同， swizzle 的行为相同。

```text
[[(0:0),(0:1),(0:2),(0:3)]  // phase 0 (xor with 0)
[ (1:1),(1:0),(1:3),(1:2)]  // phase 1 (xor with 1)
[ (2:0),(2:1),(2:2),(2:3)]  // phase 2 (xor with 0)
[ (3:1),(3:0),(3:3),(3:2)]] // phase 3 (xor with 1)
```

- #shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>

相邻 2(vec) 元素为一组，进行 xor。 `out[r][c] = in[r][(c / 2) ^ r) * 2 + (c % 2)]`

```text
[[(0:0),(0:1),(0:2),(0:3)]  // phase 0
[ (1:2),(1:3),(1:0),(1:1)]  // phase 1
[ (2:0),(2:1),(2:2),(2:3)]  // phase 2
[ (3:2),(3:3),(3:0),(3:1)]] // phase 3
```

- #shared<{vec=2, perPhase=2, maxPhase=4, order=[1,0]}>

`out[r][c] = in[r][(c / 2) ^ (r % 2)) * 2 + (c % 2)]`

```text
[[(0:0),(0:1),(0:2),(0:3)]
[ (1:0),(1:1),(1:2),(1:3)]
[ (2:2),(2:3),(2:0),(2:1)]
[ (3:2),(3:3),(3:0),(3:1)]]
```

![swizzled memory](/assets/img/blog/img_triton_survey/swizzled.png)

## distributed layout

distributed layout 使用映射函数描述整个 tensor 的访问模式。映射函数(layout function)会将特定的Tensor交给特定的Thread去处理(即一个layout描述整个tensor的访问模式)，达到一个**distribution**的效果

layout function 说明：

```cpp
- 计算公式
\mathcal{L}(A)[i_d] = L[(i_d + k_d*A.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*A.shape[d] < L.shape[d]

- 举例：A 是需要访问的数据，L是当前线程的布局
A = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

// L(i, j) = {...} 用来描述数据 (i, j) 被哪些CUDA线程访问
L(A) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
        {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]

- 计算过程
d = 0 或 1， 0 <= k_0 <= 3，0 <= k_1 <= 3，A.shape = [2, 8]， L.shape = [4, 4]
那么负责访问 A[1, 3] 的线程为：
(i_0 + k_0 * A.shape[0]) % L.shape[0] = (1 + [0, 3] * 2) % 4 = 1 或 3
(i_1 + k_1 * A.shape[1]) % L.shape[1] = (3 + [0, 3] * 8) % 4 = 3
所以负责访问 A[1, 3] 的线程是 L[1, 3] 和 L[3, 3]。
```

distributte layout 将信息分为4个维度：

- CTAs Per CGA：在 hopper 上才有用，因为 hopper 架构首次引入了 SM-to-SM
- Warps Per CTA：CTA 内 warp 的布局（对应 `warpsPerCTA`）
- Threads Per Warp：warp 内 thread 的布局（对应 `threadsPerWarp`）
- Values Per Thread：一个 thread 需要处理多少元素（对应 `sizePerThread`）

常用函数方法

```text
// 继承自 cta layout
SmallVector<unsigned> getCTAsPerCGA() const;
SmallVector<unsigned> getCTAOrder() const;
SmallVector<unsigned> getCTASplitNum() const;

SmallVector<unsigned> getWarpsPerCTA() const;
SmallVector<unsigned> getWarpOrder() const;
SmallVector<unsigned> getThreadsPerWarp() const;
SmallVector<unsigned> getThreadOrder() const;
SmallVector<unsigned> getSizePerThread() const;
// sizePerThread * threadsPerWarp * warpsPerCTA
SmallVector<unsigned> getShapePerCTATile(ArrayRef<int64_t> tensorShape = ArrayRef<int64_t>()) const;

std::optional<LinearLayout> toLinearLayout(ArrayRef<int64_t> shape) const;
```

### block layout

最常见的 layout，结合 `AxisInfoAnalysis` 获得 load 和 store 的访存行为，再用来访存合并(memory coalescing)，使得访存行为更加高效。

An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout **used to promote memory coalescing in LoadInst and StoreInst.**

- 基础概念

例如：`#triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>`

<!-- ![block_layout](/assets/img/blog/img_triton_survey/cta_warp_thread.png) -->

- sizePerThread = [1, 4]：每个线程处理的 **连续排布** 数据数目
- threadsPerWarp = [4, 8]： warp内线程的布局
- warpsPerCTA = [1, 1]：thread block内warp的布局
- order = [1, 0]：先访问dim1，再访问dim0

该BLock访存模式：每行由8个thread负责访问，每个thread会访问连续4个元素，所以一次能处理(1x4x1, 8x4x1) = (4, 32)规模的shape。如下：

```bash
$ triton-tensor-layout -l "#triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>" -t "tensor<4x32xf16>"
# T0:0,  T0:1,  T0:2,  T0:3 表示 T0 一次处理4个连续数组成的块
Print layout attribute: #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
[[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3,  T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3,  T4:0,  T4:1,  T4:2,  T4:3,  T5:0,  T5:1,  T5:2,  T5:3,  T6:0,  T6:1,  T6:2,  T6:3,  T7:0,  T7:1,  T7:2,  T7:3]
[  T8:0,  T8:1,  T8:2,  T8:3,  T9:0,  T9:1,  T9:2,  T9:3, T10:0, T10:1, T10:2, T10:3, T11:0, T11:1, T11:2, T11:3, T12:0, T12:1, T12:2, T12:3, T13:0, T13:1, T13:2, T13:3, T14:0, T14:1, T14:2, T14:3, T15:0, T15:1, T15:2, T15:3]
[ T16:0, T16:1, T16:2, T16:3, T17:0, T17:1, T17:2, T17:3, T18:0, T18:1, T18:2, T18:3, T19:0, T19:1, T19:2, T19:3, T20:0, T20:1, T20:2, T20:3, T21:0, T21:1, T21:2, T21:3, T22:0, T22:1, T22:2, T22:3, T23:0, T23:1, T23:2, T23:3]
[ T24:0, T24:1, T24:2, T24:3, T25:0, T25:1, T25:2, T25:3, T26:0, T26:1, T26:2, T26:3, T27:0, T27:1, T27:2, T27:3, T28:0, T28:1, T28:2, T28:3, T29:0, T29:1, T29:2, T29:3, T30:0, T30:1, T30:2, T30:3, T31:0, T31:1, T31:2, T31:3]]
```

但若输入op的shape为(8, 32)，那么让每个thread处理两个连续块即可，即第一个thread处理(0, 0:3), (4, 0:3)两个块。

```bash
$ triton-tensor-layout -l "#triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>" -t "tensor<8x32xf16>"
Print layout attribute: #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
[[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3,  T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3,  T4:0,  T4:1,  T4:2,  T4:3,  T5:0,  T5:1,  T5:2,  T5:3,  T6:0,  T6:1,  T6:2,  T6:3,  T7:0,  T7:1,  T7:2,  T7:3]
[  T8:0,  T8:1,  T8:2,  T8:3,  T9:0,  T9:1,  T9:2,  T9:3, T10:0, T10:1, T10:2, T10:3, T11:0, T11:1, T11:2, T11:3, T12:0, T12:1, T12:2, T12:3, T13:0, T13:1, T13:2, T13:3, T14:0, T14:1, T14:2, T14:3, T15:0, T15:1, T15:2, T15:3]
[ T16:0, T16:1, T16:2, T16:3, T17:0, T17:1, T17:2, T17:3, T18:0, T18:1, T18:2, T18:3, T19:0, T19:1, T19:2, T19:3, T20:0, T20:1, T20:2, T20:3, T21:0, T21:1, T21:2, T21:3, T22:0, T22:1, T22:2, T22:3, T23:0, T23:1, T23:2, T23:3]
[ T24:0, T24:1, T24:2, T24:3, T25:0, T25:1, T25:2, T25:3, T26:0, T26:1, T26:2, T26:3, T27:0, T27:1, T27:2, T27:3, T28:0, T28:1, T28:2, T28:3, T29:0, T29:1, T29:2, T29:3, T30:0, T30:1, T30:2, T30:3, T31:0, T31:1, T31:2, T31:3]
[  T0:4,  T0:5,  T0:6,  T0:7,  T1:4,  T1:5,  T1:6,  T1:7,  T2:4,  T2:5,  T2:6,  T2:7,  T3:4,  T3:5,  T3:6,  T3:7,  T4:4,  T4:5,  T4:6,  T4:7,  T5:4,  T5:5,  T5:6,  T5:7,  T6:4,  T6:5,  T6:6,  T6:7,  T7:4,  T7:5,  T7:6,  T7:7]
[  T8:4,  T8:5,  T8:6,  T8:7,  T9:4,  T9:5,  T9:6,  T9:7, T10:4, T10:5, T10:6, T10:7, T11:4, T11:5, T11:6, T11:7, T12:4, T12:5, T12:6, T12:7, T13:4, T13:5, T13:6, T13:7, T14:4, T14:5, T14:6, T14:7, T15:4, T15:5, T15:6, T15:7]
[ T16:4, T16:5, T16:6, T16:7, T17:4, T17:5, T17:6, T17:7, T18:4, T18:5, T18:6, T18:7, T19:4, T19:5, T19:6, T19:7, T20:4, T20:5, T20:6, T20:7, T21:4, T21:5, T21:6, T21:7, T22:4, T22:5, T22:6, T22:7, T23:4, T23:5, T23:6, T23:7]
[ T24:4, T24:5, T24:6, T24:7, T25:4, T25:5, T25:6, T25:7, T26:4, T26:5, T26:6, T26:7, T27:4, T27:5, T27:6, T27:7, T28:4, T28:5, T28:6, T28:7, T29:4, T29:5, T29:6, T29:7, T30:4, T30:5, T30:6, T30:7, T31:4, T31:5, T31:6, T31:7]]
```

当 `sizePerThread = [2, 4]` 时也可以一次处理完(8, 32)的数据，与上面那种执行两边的方法区别上，layout在 row 上会连续。

```bash
[[ T0:0,  T0:1,  T0:2,  T0:3,  T1:0,  T1:1,  T1:2,  T1:3,  T2:0,  T2:1,  T2:2,  T2:3,  T3:0,  T3:1,  T3:2,  T3:3,  T4:0,  T4:1,  T4:2,  T4:3,  T5:0,  T5:1,  T5:2,  T5:3,  T6:0,  T6:1,  T6:2,  T6:3,  T7:0,  T7:1,  T7:2,  T7:3]
[  T0:4,  T0:5,  T0:6,  T0:7,  T1:4,  T1:5,  T1:6,  T1:7,  T2:4,  T2:5,  T2:6,  T2:7,  T3:4,  T3:5,  T3:6,  T3:7,  T4:4,  T4:5,  T4:6,  T4:7,  T5:4,  T5:5,  T5:6,  T5:7,  T6:4,  T6:5,  T6:6,  T6:7,  T7:4,  T7:5,  T7:6,  T7:7]
[  T8:0,  T8:1,  T8:2,  T8:3,  T9:0,  T9:1,  T9:2,  T9:3, T10:0, T10:1, T10:2, T10:3, T11:0, T11:1, T11:2, T11:3, T12:0, T12:1, T12:2, T12:3, T13:0, T13:1, T13:2, T13:3, T14:0, T14:1, T14:2, T14:3, T15:0, T15:1, T15:2, T15:3]
...
```

- memory coalesce

一个简单的例子说明：

假如一个 warp 希望访问 128 个数(tensor<128xf16>)，下面的 layout 会通过四次搬运完成：

```text
#blocked_before = #blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
```

memory-coalesce 后将会让每个 thread 处理的数据更多，这样一次就可以搬运完成更能在后端映射成 vectorization 指令。

```text
#blocked_after = #blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
```

### MMA Layout 和 DotOperand Layout

用来指导 op 下降到特殊指令的 attr。

1.MMA Layout

表示 Tensor Core 中 MMA 指令结果的 data layout，一般可以直接对应到 PTX 指令中相应的数据排布需求。

> 和硬件相关很大，这里不展开

2.DotOperand Layout

表示 dotOp 的输入的 layout。主要包含 `opIdx` 和 `parent` 两个信息，

- opIdx ：用来标识 dotOp 的操作数
  - opIdx=0 表示 DotOp 的 $a
  - opIdx=1 表示 DotOp 的 $b
- parent：决定了 DotOperand 的布局方式
  - MMA Layout（如果 DotOp lower 到 MMA 指令）
  - Blocked Layout（如果 DotOp lower 到 FMA 指令）

## Slice Layout

Slice Layout 通过给定的 parent 布局和 dim 维度来压缩指(squeezing)定维度。

- 如果 dim = 0，则会将不同列的数据组合在一起形成新的 layout
- 如果 dim = 1，则会将不同行的数据组合在一起形成心的 layout

## linear layout

类似 CUTLASS v3 中的 CuTe Layout，在 [PR](https://github.com/triton-lang/triton/pull/3794) 中第一次引入，用于表示生成 indices 的行为。

## tools for layout

在 [PR](https://github.com/triton-lang/triton/pull/4486) 中合入了一个可以打印 ttgir 上 layout 的工具 `triton-tensor-layout`，通过调用 `getLayoutStr` 来解析 RankedTensorType 中的 layout 信息，且当前已经支持了 shared layout 的 dump。例如：

- distribute layout

```bash
$ triton-tensor-layout -l "#triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>" -t "tensor<16x16xf16>"
Print layout attribute: #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
# 每行用8个thread，每行中每个thread会负责连续的4个元素，但一行只有16个元素，所以4个thread就能遍历完一遍了
# 因此 T0 和 T4 都会访问第0行的前四个元素
[[  T0:0|  T4:0,   T0:1|  T4:1,   T0:2|  T4:2,   T0:3|  T4:3,   T1:0|  T5:0,   T1:1|  T5:1,   T1:2|  T5:2,   T1:3|  T5:3,   T2:0|  T6:0,   T2:1|  T6:1,   T2:2|  T6:2,   T2:3|  T6:3,   T3:0|  T7:0,   T3:1|  T7:1,   T3:2|  T7:2,   T3:3|  T7:3]
[   T8:0| T12:0,   T8:1| T12:1,   T8:2| T12:2,   T8:3| T12:3,   T9:0| T13:0,   T9:1| T13:1,   T9:2| T13:2,   T9:3| T13:3,  T10:0| T14:0,  T10:1| T14:1,  T10:2| T14:2,  T10:3| T14:3,  T11:0| T15:0,  T11:1| T15:1,  T11:2| T15:2,  T11:3| T15:3]
[  T16:0| T20:0,  T16:1| T20:1,  T16:2| T20:2,  T16:3| T20:3,  T17:0| T21:0,  T17:1| T21:1,  T17:2| T21:2,  T17:3| T21:3,  T18:0| T22:0,  T18:1| T22:1,  T18:2| T22:2,  T18:3| T22:3,  T19:0| T23:0,  T19:1| T23:1,  T19:2| T23:2,  T19:3| T23:3]
[  T24:0| T28:0,  T24:1| T28:1,  T24:2| T28:2,  T24:3| T28:3,  T25:0| T29:0,  T25:1| T29:1,  T25:2| T29:2,  T25:3| T29:3,  T26:0| T30:0,  T26:1| T30:1,  T26:2| T30:2,  T26:3| T30:3,  T27:0| T31:0,  T27:1| T31:1,  T27:2| T31:2,  T27:3| T31:3]
[  T32:0| T36:0,  T32:1| T36:1,  T32:2| T36:2,  T32:3| T36:3,  T33:0| T37:0,  T33:1| T37:1,  T33:2| T37:2,  T33:3| T37:3,  T34:0| T38:0,  T34:1| T38:1,  T34:2| T38:2,  T34:3| T38:3,  T35:0| T39:0,  T35:1| T39:1,  T35:2| T39:2,  T35:3| T39:3]
[  T40:0| T44:0,  T40:1| T44:1,  T40:2| T44:2,  T40:3| T44:3,  T41:0| T45:0,  T41:1| T45:1,  T41:2| T45:2,  T41:3| T45:3,  T42:0| T46:0,  T42:1| T46:1,  T42:2| T46:2,  T42:3| T46:3,  T43:0| T47:0,  T43:1| T47:1,  T43:2| T47:2,  T43:3| T47:3]
[  T48:0| T52:0,  T48:1| T52:1,  T48:2| T52:2,  T48:3| T52:3,  T49:0| T53:0,  T49:1| T53:1,  T49:2| T53:2,  T49:3| T53:3,  T50:0| T54:0,  T50:1| T54:1,  T50:2| T54:2,  T50:3| T54:3,  T51:0| T55:0,  T51:1| T55:1,  T51:2| T55:2,  T51:3| T55:3]
[  T56:0| T60:0,  T56:1| T60:1,  T56:2| T60:2,  T56:3| T60:3,  T57:0| T61:0,  T57:1| T61:1,  T57:2| T61:2,  T57:3| T61:3,  T58:0| T62:0,  T58:1| T62:1,  T58:2| T62:2,  T58:3| T62:3,  T59:0| T63:0,  T59:1| T63:1,  T59:2| T63:2,  T59:3| T63:3]
[  T64:0| T68:0,  T64:1| T68:1,  T64:2| T68:2,  T64:3| T68:3,  T65:0| T69:0,  T65:1| T69:1,  T65:2| T69:2,  T65:3| T69:3,  T66:0| T70:0,  T66:1| T70:1,  T66:2| T70:2,  T66:3| T70:3,  T67:0| T71:0,  T67:1| T71:1,  T67:2| T71:2,  T67:3| T71:3]
[  T72:0| T76:0,  T72:1| T76:1,  T72:2| T76:2,  T72:3| T76:3,  T73:0| T77:0,  T73:1| T77:1,  T73:2| T77:2,  T73:3| T77:3,  T74:0| T78:0,  T74:1| T78:1,  T74:2| T78:2,  T74:3| T78:3,  T75:0| T79:0,  T75:1| T79:1,  T75:2| T79:2,  T75:3| T79:3]
[  T80:0| T84:0,  T80:1| T84:1,  T80:2| T84:2,  T80:3| T84:3,  T81:0| T85:0,  T81:1| T85:1,  T81:2| T85:2,  T81:3| T85:3,  T82:0| T86:0,  T82:1| T86:1,  T82:2| T86:2,  T82:3| T86:3,  T83:0| T87:0,  T83:1| T87:1,  T83:2| T87:2,  T83:3| T87:3]
[  T88:0| T92:0,  T88:1| T92:1,  T88:2| T92:2,  T88:3| T92:3,  T89:0| T93:0,  T89:1| T93:1,  T89:2| T93:2,  T89:3| T93:3,  T90:0| T94:0,  T90:1| T94:1,  T90:2| T94:2,  T90:3| T94:3,  T91:0| T95:0,  T91:1| T95:1,  T91:2| T95:2,  T91:3| T95:3]
[  T96:0|T100:0,  T96:1|T100:1,  T96:2|T100:2,  T96:3|T100:3,  T97:0|T101:0,  T97:1|T101:1,  T97:2|T101:2,  T97:3|T101:3,  T98:0|T102:0,  T98:1|T102:1,  T98:2|T102:2,  T98:3|T102:3,  T99:0|T103:0,  T99:1|T103:1,  T99:2|T103:2,  T99:3|T103:3]
[ T104:0|T108:0, T104:1|T108:1, T104:2|T108:2, T104:3|T108:3, T105:0|T109:0, T105:1|T109:1, T105:2|T109:2, T105:3|T109:3, T106:0|T110:0, T106:1|T110:1, T106:2|T110:2, T106:3|T110:3, T107:0|T111:0, T107:1|T111:1, T107:2|T111:2, T107:3|T111:3]
[ T112:0|T116:0, T112:1|T116:1, T112:2|T116:2, T112:3|T116:3, T113:0|T117:0, T113:1|T117:1, T113:2|T117:2, T113:3|T117:3, T114:0|T118:0, T114:1|T118:1, T114:2|T118:2, T114:3|T118:3, T115:0|T119:0, T115:1|T119:1, T115:2|T119:2, T115:3|T119:3]
[ T120:0|T124:0, T120:1|T124:1, T120:2|T124:2, T120:3|T124:3, T121:0|T125:0, T121:1|T125:1, T121:2|T125:2, T121:3|T125:3, T122:0|T126:0, T122:1|T126:1, T122:2|T126:2, T122:3|T126:3, T123:0|T127:0, T123:1|T127:1, T123:2|T127:2, T123:3|T127:3]]
```

- shared layout

```bash
$ triton-tensor-layout -l "#triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 4, order = [1,0], hasLeadingOffset = false}>" -t "tensor<4x8xf16>"
Print layout attribute: #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
[[(0:0),(0:1),(0:2),(0:3),(0:4),(0:5),(0:6),(0:7)]
[ (1:2),(1:3),(1:0),(1:1),(1:6),(1:7),(1:4),(1:5)]
[ (2:4),(2:5),(2:6),(2:7),(2:0),(2:1),(2:2),(2:3)]
[ (3:6),(3:7),(3:4),(3:5),(3:2),(3:3),(3:0),(3:1)]]
```

默认打印是每个 thread 对应的 tensor 上的元素，默认是从 tensor 角度出发来打印每个 thread 对应的 tensor 上的元素；
也可以增加 `-use-hw-view` 以获取 warps, threads 角度获得更多信息。

用法细节请自行阅读：bin/triton-tensor-layout.cpp 以及 lib/Dialect/TritonGPU/IR/Dialect.cpp。

# ttir-2-ttgir
