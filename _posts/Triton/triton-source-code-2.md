title: OpenAI Triton 源码走读[ttir-to-ttgir]

add_convert_to_ttgpuir

`createConvertTritonToTritonGPUPass`：将 ttir 转为 triton gpu ir，需要使用来自 kernel 的 `numWarps`, `threadsPerWarp`, `numCTAs` 参数，这些信息会随着 `TritonGPUTypeConverter` 在 conversion 的过程中被用到。

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

用来描述不同 thread 如何同时访问一个位于 shared memory 上的 tensor。主要有 `vec`、`perPhase`、`maxPhase`、`order` 四个属性。用来指导如何进行 swizzling。

swizzle：调整数据在 shared memory 上的存储位置，保证 thread 访问时不出现 bank conflict。

- vec：同行的元素进行swizzle时，连续vec个为一组
- perPhase：一次 swizzling phase 处理的 row 数。连续的 perPhase 行用相同的swizzle方法。
- maxPhase：swizzling phase的个数

最基础的方法就是地址 与 phase_id 进行 xor。

假设现在有4x4个元素需要存到 shared memory 上去，存放的初始地址为：

```text
  [ 0,  1,  2,  3],
  [ 4,  5,  6,  7],
  [ 8,  9, 10, 11],
  [12, 13, 14, 15]
```

- #shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>

不同行中的地址与不同的参数 xor。

```text
  [ 0,  1,  2,  3],  // xor with 0
  [ 5,  4,  7,  6],  // xor with 1
  [10, 11,  8,  9],  // xor with 2
  [15, 14, 13, 12]   // xor with 3
```

- #shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>

相邻两行中的地址用相同的参数xor。

```text
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14]
```

- #shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>

```text
  [ 0,  1,  2,  3,  4,  5,  6,  7],
  [10, 11,  8,  9, 14, 15, 12, 13],
  [20, 21, 22, 23, 16, 17, 18, 19],
  [30, 31, 28, 29, 26, 27, 24, 25]
```

<!-- ![swizzled memory](/assets/img/blog/img_triton_survey/swizzled.png) -->

## distributed layout

映射函数(layout function)会将特定的Tensor交给特定的Thread去处理(即一个layout描述整个tensor的访问模式)，达到一个**distribution**的效果

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

### block layout

最常见的 layout，包含了配合 AxisInfoAnalysis 分析获得 load 和 store 的访存行为，以用来访存合并。

> 一个 warp 中的所有 thread 在同一时间点只能执行相同的指令，所以需要访问的内存越连续，最后 load/store transactions 的数量就越少。配合 shared layout 来调整数据分布，减少 transactions。

An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout **used to promote memory coalescing in LoadInst and StoreInst.**

`#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>`

<img src="/assets/img/blog/img_triton_survey/cta_warp_thread.png" alt="Untitled" style="zoom:50%;" />

- sizePerThread = [1, 8]：每个线程处理数据Size
- threadsPerWarp = [8, 4]： warp内线程的布局
- warpsPerCTA = [8, 1]：thread block内warp的布局
- order = [1, 0]：先访问dim1，再访问dim0

> Triton 会优先 `Contiguity` 更大的维度， `Contiguity`  信息一般来自于使用 `tl.max_contiguous(input, values)` 人为告知编译器，这意味着 input[i] 中每 values[i] 个相邻元素是连续的。

该BLock访存模式一次能处理(1x8x8, 8x4) = (64, 32)规模的shape。但若输入op的shape为(128, 32)，那么让每个thread处理两个连续块即可，即第一个thread处理(0, 0:7), (64, 0:7)两个块

假如一个 warp 希望访问 128 个数，32 个 thread 可以通过四次搬运完成：

```text
#blocked_before = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
```

memory-coalesce 后将会生成下面的 Layout(第二维连续更长，所以order也要跟着改变)，这样每个 thread 处理的数据更多，更能在后端映射成 vectorization 指令。

```text
#blocked_after = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

### MMA Layout 和 DotOperand Layout

用来指导 op 下降到特殊指令的 attr。



## linear layout

类似 CUTLASS v3 中的 CuTe Layout，在 [PR](https://github.com/triton-lang/triton/pull/3794) 中第一次引入，用于表示生成 indices 的行为。
