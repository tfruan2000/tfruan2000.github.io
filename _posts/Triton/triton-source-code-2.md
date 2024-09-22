title: OpenAI Triton 源码走读[ttir-to-ttgir]

add_convert_to_ttgpuir

`createConvertTritonToTritonGPUPass`：将 ttir 转为 triton gpu ir，需要使用来自 kernel 的 `numWarps`, `threadsPerWarp`, `numCTAs` 参数，这些信息会随着 `TritonGPUTypeConverter` 在 conversion 的过程中被用到。

# layout

自 [[RFC] To generalize TritonGPU dialect for GPU of different vendors](https://github.com/triton-lang/triton/issues/2639) 后， TritonGPU Dialect 中的 Layout Attribute 正式统一为下图

![TritonGPU Attr](/assets/img/blog/img_triton_pass/layout_attr.png)

Layout：定义了Data是如何被Thread处理。这种layout attr在lowering过程中被传递，用于描述op的拆分映射关系

> Block layout等 distributed layout 描述线程的访存行为；shared layout 描述smem中哪些元素会被同时访问，然后进行swizzled，防止banck conflict

- **Distributed Layout：**Blocked Layout, MMA Layout, DotOperand Layout都属于此类。这些Layout的特点都是映射函数会将特定的Tensor交给特定的Thread去处理(即一个layout描述整个tensor的访问模式)，达到一个**distribution**的效果

```cpp
class DistributedEncoding<string name> : TritonGPU_Attr<name> {
  let description =[{
    The layout function \mathcal{L} of this layout is then defined, for an
    index `i` \in R^D, as follows:

    \mathcal{L}(A)[i_d] = L[(i_d + k_d*A.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*A.shape[d] < L.shape[d]

    For example, for a tensor/layout pair
    A = [x  x  x  x  x  x  x  x]
        [x  x  x  x  x  x  x  x]
    L = [0  1  2  3 ]
        [4  5  6  7 ]
        [8  9  10 11]
        [12 13 14 15]

    Then the data of A would be distributed as follow between the 16 CUDA threads:
    // L(i, j) = {...} 用来描述数据 (i, j) 被哪些CUDA线程访问
    L(A) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
            {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
    }];
...
}
```

- **Shared Layout**: GPU中的Shared Memory是可以被一个Block内的任意线程访问的，shared layout会被用来描述哪些元素会被线程同时访问，以此来减少bank confict映射函数被定义为任意Tensor->任意Thread。

## distributed layout

Distributed encodings have a layout function that is entirely characterized by a d-dimensional tensor L. Note that L doesn't need to have the same shape (or even the same rank) as the tensor it is encoding.

映射函数(layout function)会将特定的Tensor交给特定的Thread去处理(即一个layout描述整个tensor的访问模式)，达到一个**distribution**的效果

![distribute_layout](/assets/img/blog/img_triton_survey/distribute_layout.png)

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

## shared layout

In order to **avoid shared memory bank conflicts**, elements may be **swizzled** in memory.

同一个warp内的thread同时访问同一列的数据会产生 bank 冲突，对数据进行 swizzle，调整相关的存储位置，保证 thread 访问时不出现 bank conflict。

![swizzled memory](/assets/img/blog/img_triton_survey/swizzled.png)
