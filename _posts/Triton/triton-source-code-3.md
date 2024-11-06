title: OpenAI Triton 源码走读[transforms in ttgir]

# 前言

在 `ttgir` 的 transforms 中，最基础的是 新增 op、 layout、analysis 三者，利用好这三者来完成变换。

1.op

`ttgir` 表示和硬件相关的计算表示。主要新增的 op 例如：

- alloc_tensor : tensor<128x32xf32, #share> ：申请shared memory
- insert_slice_async： 往 shared memory 上 insert 一个 slice，语意上类似 `tensor.insert_slice`，底层对应的cp.async指令
- async_commit_group： 底层对应的 cp.async.commit_group 指令，就是将前面的所有 cp.async 指令打包到一起执行
- async_wait {num}： 底层对应的 cp.async.wait_group 指令，也就是等待前面 num 个 cp.async-groups 执行完
- cnovert_layout： 改变 tensor 的 layout

2.layout

在 [ttir-2-ttgir](https://tfruan2000.github.io/posts/triton-source-code-2/#layout) 一文中有详细介绍。简单来说：

- shared layout： 表示一种 tensor 编码模式，用于指导如何进行 `swizzle`，使得不同 thread 在访问 tensor(on shared memory) 上的元素时尽量避免 bank conflict。
- distributed layout： 使用映射函数描述整个 tensor 的访问模式。映射函数(layout function)会将特定的Tensor交给特定的Thread去处理(即一个layout描述整个tensor的访问模式)，达到一个**distribution**的效果。
  - 最常见的 layout，结合 `AxisInfoAnalysis` 获得 load 和 store 的访存行为，再用来访存合并(memory coalescing)，使得访存行为更加高效。
  - Slice Layout 通过给定的 parent layout 和 dim 维度来压缩指(squeezing)定维度。(在 `tt.expand_dim` 的 ttir-to-ttgir 转换中使用，parent layout 表示转换的 target layout)。
  - MMA Layout 和 DotOperand Layout：指导 op 下降行为的 attr。

3.Analysis

见下文。

# Analysis

## AxisInfo

```bash
include/triton/Analysis/AxisInfo.h
lib/Analysis/AxisInfo.cpp
```

`AxisInfo` 会对 load、store 等对指针进行操作的 op 及相关 op 进行跟踪，分析出 `divisibility, contiguity, constancy` 信息，从而辅助之后的 pass 进行，以获得高 IO 效率。

1.元素

- divisibility：维度i上，所有元素的最大二次幂公约数。该值用来表示指针地址的对齐，指导后续访存操作的下降
- contiguity：维度i上，连续元素的最小长度，说明至少每contiguity[i]个元素是连续的
- constancy：维度i上，重复元素的最小长度，说明至少每constancy[i]个元素是重复的
- constantValue：表示值为常数，一般信息源头来自 `arith.constant`

以下面两种数据为例，上述三种信息的值为：

```bash
[[10, 11, 12, 13, 18, 19, 20, 21],
 [20, 21, 22, 23, 28, 29, 30, 31]]
- divisibility: [1, 2]
- stride: [4, 2]
- strideValue: [1, 10]

[[12, 16, 20, 24],
 [13, 17, 21, 25],
 [14, 18, 22, 26],
 [15, 19, 23, 27]]
- divisibility: [4, 1]
- stride: [4, 4]
- strideValue: [1, 4]
```

若要从某个地址上取 `tensor<8x32xi32>` 的数据，分析出 divisibility = [1, 1], stride = [8, 32], strideValue = [1, 0]，
这说明：

- shape[0] = 8 & stride[0] = 8 & strideValue[0] = 1 -> 第 0 维上每 8 个数据连续
- shape[1] = 32 & stride[1] = 32 & strideValue[1] = 0 -> 第 1 维上的每 32 个数据相同

2.作用

利用这些信息可以优化放存行为：

- 当 load 16x256 个数时，若低维每 256 个数重复，那么可以只 load 16 个数据，再 broadcast 成 16x256。
- mask全为true或false时可以转为 scf.if + load，最次也可以优化为 `select %mask, %load, %other`

3.打印方法

可以通过 `test-print-alignment` pass 来打印 `AnisInfo`，详见 [TestAxisInfo](https://github.com/triton-lang/triton/blob/main/test/lib/Analysis/TestAxisInfo.cpp) 以及 test/Analysis/test-alignment.mlir。

4.ModuleAxisInfoAnalysis

为 module 内每个 `FunctionOpInterface` 为单元收集 `DenseMap<Value, AxisInfo>`，这里假设了所有 func 之间都是独立的，没有 `func.call` 这种互相调用的行为(提前 inline 也行)。

每个 func 维护一个信息 `DenseMap<Value, AxisInfo>`，其中 Value 包含 func 内所有 `OpResult` 以及 `BlockArguement`。如下：

```cpp
  auto updateAxisInfoMap = [&](Value value) {
    auto axisInfo = analysis->getLatticeElement(value)->getValue();
    AxisInfo curAxisInfo;
    if (axisInfoMap->count(value)) {
      curAxisInfo = AxisInfo::join(axisInfo, axisInfoMap->lookup(value));
    } else {
      curAxisInfo = axisInfo;
    }
    (*axisInfoMap)[value] = curAxisInfo;
  };
```

当重复访问到已存在 Value 时会进行 Lattice 的交汇join，一般Lattice的join方法返回的是表示是否发生改变的状态信息 `ChangeResult`，但 AxisInfo 中重写了相关的 join 方法：

- lhs/rhs的rank为0时，意味着其中一个AxisInfo对象没有初始化或为空，直接返回另一个对象
- 循环遍历每个维度d，计算contiguity、divisibility和constancy的最小公倍数(gcd)

  ```cpp
  contiguity.push_back(gcd(lhs.getContiguity(d), rhs.getContiguity(d)));
  divisibility.push_back(gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
  constancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
  ```

- 对于 `constantValue`，当且仅当 lhs 和 rhs 的 constantValue 都存在且相等时才保留，反之都为 `std::nullopt_t`。

> 更多mlir数据流相关的内容请见[dataflow-analysis](https://tfruan2000.github.io/posts/mlir-code-note/#dataflow-analysis)。

# Transforms

```bash
include/triton/Dialect/TritonGPU/Transforms/
lib/Dialect/TritonGPU/Transforms/
```

按 `python/src/passes.cc` 中组织的 transforms pipeline：

> 感觉 ttgir 的这些 pass 格式上没有 ttir 中的易读，因为在 [PR3971](https://github.com/triton-lang/triton/pull/3971/files#diff-d607223a38c42aeeb2b41abb0fa3bfd2496ffe31858d541181d0f41c36672217R11) 中使用了
>
> 关于一个 pass 大致的格式可以看官方实现或者 [如何添加一个pass](https://tfruan2000.github.io/posts/mlir-code-note/#%E5%86%99%E4%B8%80%E4%B8%AA-pass)。

## add_coalesce

`createTritonGPUCoalesce`: 调整访存相关 op 的 ptr layout

### 代码逻辑

1.收集输入module的 `ModuleAxisInfoAnalysis`(以 func 为单位，收集func内每个value的信息)

2.遍历每个 module 内每个 memory access op，只有当其 ptr value 的 Type 是 RankedTensorType 并且 elemType 是 PointerType 时(例如tensor<256x!tt.ptr<f32>>)才继续处理。

3.为每个符合上述条件的op setCoalescedEncoding：

- `getShapePerCTA(ptr.getType())` 获得的一般就是 ptr type 的 shape，因为 default layout 中的默认 CTAsPerCGA = CTASplitNum = [1...1]，表示每个CTA是独立的。
- 使用 `multiRootGetSlice` 收集当前 op 所有相关的 op，返回这些 op 的拓扑排序。这个函数还挺有意思，使用 `getBackwardSlice` 和 `forwardSlice` 分别收集相关的 producer op 和 consumer op。直到worklist不再增长。
- 遍历上述收集到的 `SetVector`，将 shape 相同且 order 相同的 memory access op 加入 memAccessesSameOrder 中。
- 遍历 `memAccessesSameOrder` 中的 op，获得每个 op 的 `getNumElementsPerThread` 函数输出(表示可以选择的最大perThread)，一直维护一个最大值；最大的 `perThread` 再取和 `numElems / numThreads` 比较的较小值。
- 对于能对global memory写的memory access op，perThread 为了性能考量需要尽量至少128bits对齐，所以有 `min(alignment, 128 / elemNumBits)` 的逻辑。
- 采用得到的新 `perThread` 作为 `sizePerThread[order[0]]` 构建新的 BlockLayout。

```text
$ triton-opt -tritongpu-coalesce tmp.mlir --debug-only=tritongpu-coalesce
[tritongpu-coalesce]: Considering op: %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
[tritongpu-coalesce]: axis info of pointer: contiguity = [1024], divisibility = [16], constancy = [1], constant_value = <none>
[tritongpu-coalesce]: order=[0]
[tritongpu-coalesce]: shapePerCTA=[1024]
[tritongpu-coalesce]: perThread for op: 4
[tritongpu-coalesce]: perThread: 4

#blocked 是 loadOp coalesce 前的 layout， #blocked 是 loadOp coalesce 后的 layout。
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

4.

## add_optimize_thread_locality

`createTritonGPUOptimizeThreadLocality`:

## add_pipeline

`createTritonGPUPipeline`: 主要针对 DotOp 进行 global memory 到 shared memory 的数据拷贝，并做 Double Buffer 或者 N Buffer 的优化。

## add_prefetch

`createTritonGPUPrefetch`：类似 pipeline pass，也是 Double buffer 和 N Buffer 的优化，区别是做 shared memory 到 register 的数据搬运。

## add_accelerate_matmul

`createTritonGPUAccelerateMatmul`

## add_reorder_instructions

`createTritonGPUReorderInstructions`

## add_f32_dot_tc

`createTritonGPUF32DotTC`

## add_optimize_dot_operands

`createTritonGPUOptimizeDotOperands`

## add_remove_layout_conversions

`createTritonGPURemoveLayoutConversions`

## add_reduce_data_duplication

`createTritonGPUReduceDataDuplication`

## add_allocate_shared_memory

`createAllocateSharedMemoryPass`

## add_combine_tensor_select_and_if

`createTritonGPUCombineTensorSelectAndIf`: 当 `arith.select` 和 `scf.if` 的 cond 相同时，将它们结合起来，大致是以下的行为。

```text
%select = arith.select %cond, %trueVal, %falseVal
%if = scf.if %cond -> (...) {
  ...
  scf.yield ...
} else {
  scf.yield ...
}
use %select

->

%select = arith.select %cond, %trueVal, %falseVal
%if:2 = scf.if %cond -> (...) {
  ...
  scf.yield ..., %trueVal
} else {
  scf.yield ..., %falseVal
}
use %if#1
```

> 这算是一个 Arith 上的通用 transform，和 TritonGPU 没啥关系(感觉没必要放在 TritonGPU/Transforms)下，可以多次调用
>
> 当前调用点的位置也有点奇怪。

### 代码逻辑

(1)先标准化在 user of select in scf.if，因为后续用 `dom.dominates(ifOp, user)` 判断 ifOp 对 user 的支配关系时，当 userOp 在 ifOp 内部时也会返回 true。
因此先对 arith.select 进行标准化，替换掉特定的 user。减少重复判断。

> 也可以改用 `dom.properlyDominates(ifOp, user, false)` 过滤掉包含的影响。

(2)遍历 arith.select，找其 cond 的 user，若是 scf.if 且符合一定条件(arith.select的res的users都被scf.if dominates)收集，收集在 `MapVector<scf::IfOp, SmallVector<arith::SelectOp>>` 中。

(3)然后遍历收集的 `MapVector`，为每一组都创建一个新的 scf.if，arith.select的trueVal 和 falseVal 分别作为 thenRegion 的返回值和 elseRegion 的返回值，然后替换 arith.select的res的users。

### nits

增强鲁棒性的优化：scf.if 的 cond 和 arith.select 的 cond 有**取反**关系，可以将 arith.select 的 trueVal 和 falseVal 分别作为 elseRegion 的返回值和 thenRegion 的返回值

```text
%select = arith.select %cond, %trueVal, %falseVal
%true = arith.constant true : i1
%newCond = arith.xori %cond, %true : i1
%if = scf.if %newCond -> (...) {
  ...
  scf.yield ...
} else {
  scf.yield ...
}
use %select
```

## add_optimize_accumulator_init

`createTritonGPUOptimizeAccumulatorInit`
