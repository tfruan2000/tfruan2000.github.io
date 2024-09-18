---
title: Triton Survey
author: tfruan
date: 2024-04-11 20:00:00 +0800
categories: [Triton]
tags: [Triton, Survey]
---

# background

## cuda vs triton

cuda和triton编程模式对比

![cuda_vs_triton](/assets/img/blog/img_triton_survey/cuda_vs_triton.png)

相比 cuda， triton 其实是 **性能和效率的 trade-off**。很多 cuda 上人为的操作行为都是交给 compiler 来自动完成，很适合对硬件了解较少的同学来实现新的模型算法。

比CUDA的SIMT编程范式，由多个thread并行处理，triton更接近SIMD编程范式，一次处理一片数据（基于block算法的编程范式）

直接对线程块进行编程，每一个操作都是应用在块上，不再控制单个的线程，省去线程之间的同步等操作

![cuda_triton](/assets/img/blog/img_triton_survey/cuda_triton.png)

## 推荐repo

OpenAI [Triton](https://github.com/triton-lang/triton/tree/main) 是什么？这个问题很多大佬都已经回答过了，阅读完以下 blog 相信大家会有个基础的理解

- 杨军老师的 [谈谈对OpenAI Triton的一些理解](https://zhuanlan.zhihu.com/p/613244988)，帮助大家建立一个宏观印象
- 董鑫大佬的 [如何入门 OpenAI Triton 编程?](https://www.zhihu.com/question/622685131/answer/3217107882)，帮助了解更多关于语法上的信息
- BobHuang大佬的 [浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)，帮助初学者大致明白一段 python 程序如何进入 triton pipeline，然后跑起来。
- 理解triton语法的repo：[triton-puzzles](https://github.com/srush/Triton-Puzzles)
- 很多用triton实现的kernel的repo：[lightllm](https://github.com/ModelTC/lightllm)

# 组成

triton 的[组成](https://github.com/triton-lang/triton/tree/main/python/triton)：

- [language](https://github.com/triton-lang/triton/tree/main/python/triton/language)：前端的triton-lang语法
- [compiler](https://github.com/triton-lang/triton/tree/main/python/triton/compiler)：kernel编译的行为，会根据后端调用对应的pipeline，把triton-lang下降到硬件汇编
- [runtime](https://github.com/triton-lang/triton/tree/main/python/triton/runtime)：cache，auto-tuning， jit 等组件
- [backends](https://github.com/triton-lang/triton/tree/main/python/triton/backends): 编译完后，在执行时掉用的是真实后端，例如 [nvgpu](https://github.com/triton-lang/triton/tree/main/third_party/nvidia/backend)，这里主要包含 compiler 时的 pipeline 组织， launch 函数(使用模版加真实kernel参数生成真实的launch函数)等

## languague

作为 triton 项目的 fontend，包含每个原语的构造行为（例如op的某些类型会偷偷转为默认的f32来计算）。

官方提供了原语的用法手册：<https://triton-lang.org/main/index.html>

**tl规定了ir必须表达为static的**，不允许 mlir 中 dynamic shape 的表达。

## compiler

以 nvgpu 为例，triton-lang 的下降流程：

triton-lang -> triton dialect -> triton gpu dialect -> nvgpu dialect -> llvm ir + nvvm ir -> ptx -> cnbin

![triton_arch_now](/assets/img/blog/img_triton_survey/triton_arch_now.png)

triton-lang(python) -> ast -> ttir：遍历整个语法树，通过code_generator相关代码，定义visit_Assign等函数，通过mlir builder生成对应的mlir。

compiler支持多后端的方向：通过Linalg dialect

![triton_arch](/assets/img/blog/img_triton_survey/triton_arch.png)

## runtime

### jit

`@triton.jit`装饰器 表示下面这段代码是一个 triton kernel

编译流程中，首先会将 `jit` 修饰的 kernel 给打包成一个 `JITFunction`，见 [runtime/jit.py](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py#L824)。

```python
def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[JITFunction[T], Callable[[T], JITFunction[T]]]:
    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        if os.getenv("TRITON_INTERPRET", "0") == "1":
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn) # 使用Interpreter执行，完全不会走到任何编译流程，使用numpy&torch api 封装
        else:
            return JITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            ) # 编译
```

> `@triton.jit` 的 `do_not_specialize` 参数，来阻止 triton 生成过多的 kernel。
> triton jit 会以每一个非指针参数为准，去生成一个kernel，比如某一个参数运行时取值可能为1或0，那么 triton 就会为它们各生成一个。

`JITFunction` 对象继承自 `KernelInterface`，并封装了一些调用语法糖。

```python
# python/triton/runtime/jit.py
class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)

class JITFunction(KernelInterface[T]):
    cache_hook = None
    ...
    def run(self, *args, grid, warmup, **kwargs):
        ...
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            # Kernel is not cached; we have to compile.
            ...
            src = self.ASTSource(self, signature, constants, configs[0]) # ast 转 ttir
            kernel = self.compile( # ttir 一路下降到 cnbin
                src,
                target=target,
                options=options.__dict__,
            )
            ...
        if not warmup:
            ...
            kernel.run(...) # 调用 drive api执行cnbin
```

`lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)`：将grid参数封装传入进去, 除了定义的kernel参数外，还会额外传入num_wraps， stages, stream等参数。

```python
# 人为写出的kernel调用
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

# 真实的kernel调用
add_kernel(x.data_ptr, y.data_ptr, output, n_elements, BLOCK_SIZE=1024, grid, num_warps=4, num_stages=3, extern_libs=None, stream=None, warmup=False)
```

### auto-tuning

@[auto-tuning](https://triton-lang.org/main/python-api/generated/triton.autotune.html) ：由 `@triton.jit`装饰的kernel可以调用 `@auto-tuning` detector触发自动调优

使用上需要提供一个configs（包含在kernel中定义的 `tl.constexpr`）列表，autotune会多次运行kernel函数来评估configs中的所有配置。（配置是人为给出的，所以空间不大，依赖人为经验）

- key：参数列表，当key中的参数改变时，需要重新评估configs

- prune_configs_by：用户可以传入函数来帮助减枝（例如基于性能模型的函数），加快收敛

- reset_to_zero：输入参数名列表，在运行前将这些参数重置为0

- warmup：每个config的warmup时间，默认25ms

- rep：每个config的重复时间，默认100ns

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
```

当前**要求所有BLOCK_SIZE设置的值都得是2次幂**，因为在gpu上数据为2次冪的规模时性能更好。

```python
n_rows, n_cols = x.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

### cache

> `triton cache` 默认存在 `~/.triton/cache/` 文件夹下，当然也可以使用 `export TRITON_CACHE_DIR=xxx` 来指定。

triton 是 jit 的执行模式，但为了减少编译时间，其实会保留每次编译得到的 kernel，所以真实的编译流程是：

1.根据 kernel 信息生成一个 `metadata_filename`(一个 json 文件)，然后在cache目录中查找

```python
    if not always_compile and metadata_path is not None:
        # cache hit!
        metadata = json.loads(Path(metadata_path).read_text())
        return CompiledKernel(src, metadata_group, hash) # hash
```

2.如果找到了就用，没找到就需要编译

python->ast->ttir->...

3.某个op多次执行，什么时候会hit cache，什么时候需要重新编译呢？

这需要从 triton 何时会产生一个新 cache 讲起。 triton 会以 [key](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py#L616) 为核心，key 包含 `sig_and_spec`, `constexpr_vals` 和 `excess_kwargs`。

- `sig_and_spec`： 函数名 和 参数类型，直接表现在 kernel 上。当参数类型很多的时候也可以使用 `do_not_specialize` 来排除掉某些参数的影响，来避免生成更多的 kernel。
- `constexpr_vals`： 标记为 `tl.constexpr` 的参数
- `excess_kwargs`：`num_stages`, `num_warps`, `num_stages` 等

## backend

### launch

由于每个 kernel 的参数可能不同，所以需要为其生成不同的执行函数，会使用 kernel 的一些信息和[固定的代码拼接](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py#L147)。最终变成一个可以调以调用的接口。

# elements

这里只是简单介绍下，举个🌰，vector add

```python
import torch
import triton
import triton.language as tl

# 这是kernel
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

# 这是launch函数
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```

## input

- pointer

`x_ptr, y_ptr` 指针，为其代表的tensor的第一个元素的地址。用来将数据load到memory

- hyper-parameter

超参数 `tl.constexptr` ，运行时传入的值来自 compiler 搜索得到。对于不同的硬件使用时，最佳性能（访存+计算）的参数可能是不同的，后续由 Triton compiler 来进行搜索不同的值

- stride

输入中一般也有stride，对于n维的tensor a，a.stride()会输出一个n维数组。stride用来找每个元素的指针

## pid

一个 job 被分成 grid 个 task。`pid = tl.program_id(axis=0)` 表示当前是第几个task

1. program_id是这个虚拟的**for 循环 里面的 index** (第几次循环，实际中这些循环是并行)

```python
pid = tl.program_id(axis=0)
# 当访问数据总长256, BLOCK_SIZE=64
# tl.arange(0, BLOCK_SIZE) -> [0, 63]
# 0， 64， 128， 192
block_start = pid * BLOCK_SIZE
# 所以数据访问时是按照 [0:64, 64:128, 128:192, 192:256]
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

2. `axis` , 是说明 "循环"有几层，此处 axis = 0表示展开为1维来访问（维度概念类比memref的维度，第一维相当于memref的最内维u）

axis是启动3d Grid的索引，必须是0 / 1 / 2

## load & store

显示地load和store，批量数据处理，以BLOCK为单位，一次处理一个BLOCK_SIZE的数据，SIMD行为

load：从DRAM读到SRAM；store：从SRAM写回DRAM；减少了DRAM上计算行为。load和store在语义上可以解释为gather和scatter

```python
# load 和 store 时都是使用基地址加偏移 获得一片数据，mask表示只获得这片数据中的一部分
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
# 写回时也需要mask
tl.store(output_ptr + offsets, output, mask=mask)
```

- offsets

offsets 为要处理数据的范围，由当前block_start和range计算而成

```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

- mask

mask 为遮盖，类似decoder Attn中的mask。一是规范访存行为，防止越界（最后一块数据大小可能不满足防止 BLOCK_SIZE 的大小）；二是过滤对本次计算不必须的数据。

例如offset=1024，mask为一个1024维的数组，每个数为0/1，当某位为1时，则load该数据，当某位为0时，舍弃。

## grid

调用kernel时，需要说明该kernel执行循环有几层，每层有几次，这就是 `grid` 的概念

以Matmul而言，若A为MxK，B为KxN，那么C的大小就是MxN（M和N为parallel axis大小，K为reduction轴大小）

每次分块计算，单块大小BLOCK_SIZE_M x BLOCK_SIZE_N，总共进行
$$
\frac{M}{\text{BLOCK\_{SIZE}\_{M}}} \times \frac{N}{\text{BLOCK\_{SIZE}\_{N}}}
$$
Triton中关于grid定义：

```python
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
```

对比Cuda中launch kernel的行为

```cpp
  dim3 block(BLOCK_SIZE_M, BLOCK_SIZE_N);
  dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
  matmul_kernel<<<grid,block>>>(Ad, Bd, Cd, M, N, K);
```

## num_warp

一般体现在module Attr上，下面的代码意味着这个程序使用4个warp执行（这个参数一般也是 `tl.constexpr`）

```python
"triton_gpu.num-warps" = 4 : i32
```

tritongpu ir相比ttir仅多了一个Blocked Layout，本质上描述的是Block对Memory的Access Pattern

```python
 #blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

就是**一个CTA里有4个Warp，一个Warp有32个Thread，一个Thread处理1个元素**。

Blocked Layout只是一种Pattern，但**按照这个Pattern会多次访问，总访问量达到BLOCK_SIZE**

## num_stages

compiler在软件流水时使用。软件流水优化一般会在kernel中插入循环，以实现对一个 `BLOCK_SIZE` 的数据进行分段计算

## layout

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

### distributed layout

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

### shared layout

In order to **avoid shared memory bank conflicts**, elements may be **swizzled** in memory.

同一个warp内的thread同时访问同一列的数据会产生 bank 冲突，对数据进行 swizzle，调整相关的存储位置，保证 thread 访问时不出现 bank conflict。

![swizzled memory](/assets/img/blog/img_triton_survey/swizzled.png)

### MMA Layout 和 DotOperand Layout

用来指导 op 下降到特殊指令的 attr。

# trick

## 不同的环境变量

- `MLIR_ENABLE_DUMP=1`

dumps the IR before every MLIR pass Triton runs

- `TRITON_PRINT_AUTOTUNING=1`

打印每次选择的 config

## 打印pass前后ir

- 使用 `triton-opt` 直接跑 pipeline，加上 `mlir-print-ir-after-all`
- 改 python 中 triton 库 `site-packages/triton/backend/xxx/compiler.py` 中的代码，例如注释掉下面某个pass来看ir是否不同

```python
    # /usr/lib/python3.10/site-packages/triton/backends/xxx/compiler.py
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod
```
