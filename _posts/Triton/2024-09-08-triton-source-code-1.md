---
title: OpenAI Triton 源码走读[transforms in ttir]
author: tfruan
date: 2024-09-08 12:00:00 +0800
categories: [Triton]
tags: [MLIR, Triton]
---

**本文将从一个 MLIR Programer 的角度来读 Triton 源码。因为是主要阅读源码，所以比较枯燥，请选择避坑～**

# 前言

OpenAI [Triton](https://github.com/triton-lang/triton/tree/main) 是什么？这个问题很多大佬都已经回答过了，阅读完以下 blog 相信大家会有个基础的理解。

- 杨军老师的 [谈谈对OpenAI Triton的一些理解](https://zhuanlan.zhihu.com/p/613244988)，帮助大家建立一个宏观印象
- 董鑫大佬的 [如何入门 OpenAI Triton 编程?](https://www.zhihu.com/question/622685131/answer/3217107882)，帮助了解更多关于语法上的信息
- BobHuang大佬的 [浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)，帮助初学者大致明白一段 python 程序如何进入 triton pipeline，然后跑起来。

有一些基础的认识后，大家就可以在 Triton 中各取所需了。作为一个 MLIR Programer，我还希望了解每个 transform pass 和 每一步 conversion 做了什么事情，所以本文是个人读源码的一个记录(枯燥预警!!)。

> 当前 Triton 中关于 arch 和 non-arch 的抽象界限比较模糊，所以 transform 和 conversion 中可能含有很多 hardware-special 的信息，所以文中会夹带我对 arch 的理解作为源码解读的补充，若有不对，烦请指正。
{: .prompt-info }

下文皆以 NVGPU 为 target 去说明 triton 的相关信息。本文基于 triton 版本：[3104056](https://github.com/triton-lang/triton/tree/310405647df51a909943bed71c5a6fd9a3e402b4)。**本文并不严格follow triton ir 下降的步骤。** 后续版本可能发生了 pass 改名、增加/删除、实现修改，请自行核对。

> 笔者也跟着一起修改了代码，加测试玩，所在分支[blog-triton-pass](https://github.com/tfruan2000/triton/tree/blog-triton-pass)

那么，这个大的项目我应该从何看起呢，按照 `triton/third_party/nvidia/backend/compiler.py` 中的代码， triton 编译流程从 triton-lang 输入算起，一共会下降到 5 个 stage：

triton-lang -> triton ir(描述上层计算) -> triton gpu ir(为tensor分配layout，表达CTA内thread的访存行为) -> llvm ir(nvvm ir) -> ptx -> cubin

以 `make_ttir` 为例，对输入的 `mod` (由 python ast转换来的最初ttir)施加一定的 transform 和 conversion pass。

```python
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
```

其中的各种 pass 的定义在 `triton/python/src/passes.cc` 文件中，大致相当于在 MLIR 中组织 Pass 的形式。

```cpp
// OpPassManager &pm
pm.addPass(createCanonicalizerPass());
pm.addPass(createCSEPass());
```

`triton/python/src/passes.cc` 文件中的 pass 组织形式很不错，那我们就从这开始发车！

# commom

## inliner

`createInlinerPass`

inliner pass 能将 function call 给 inline 到主 kernel 中来(不用散落在不同的 func 中)，方便后序代码的优化。

> 关于 inliner 在 MLIR 中的信息可见：[dialectinlinerinterface](https://tfruan2000.github.io/posts/mlir-code-note/#dialectinlinerinterface)。

## canonicalizer

`createCanonicalizerPass`

和 inliner 一样，也是 每个 dialect 都需要支持的， inliner 需要继承 `DialectInlinerInterface`，而 `canonicalizer` 需要针对 op 写 fold pattern。要为 op 加 fold pattern 还需要在 op 的 td 中增加 `let hasCanonicalizer = 1;`，这样会自动为算子生成 `getCanonicalizationPatterns` 接口。

例如，下面就是 `IfOp` 的 canoncalizePattern，有很多种花样去 fold scf.if。

```cpp
// mlir/lib/Dialect/SCF/IR/SCF.cpp
void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<CombineIfs, CombineNestedIfs, ConditionPropagation,
              ConvertTrivialIfToSelect, RemoveEmptyElseBranch,
              RemoveStaticCondition, RemoveUnusedResults,
              ReplaceIfYieldWithConditionOrValue>(context);
}
```

canonicalize 的准则并不是性能优化，而是为了后续 ir 分析更加高效。所以我经常用 canonicalize 把一些 constant 的 arith 计算给 fold 掉，减短一些 ir 链。

## cse, dce, sccp

- cse -> createCSEPass
- symbol_dce -> createSymbolDCEPass
  - 消除 dead symbal
  - symbol 通过了一种非 SSA 机制的引用方法，可以通过名称引用，在 ir 中存在函数调用时十分重要，但是 inliner 后就需要清除相关信息。
- sccp -> createSCCPPass (constant propagation)
  - 依赖 `SparseForwardDataFlowAnalysis`，最终实现将 operand value 中的 constant value 替换进去，可以帮助后续 fold 行为以及 DCE 分析。

编译优化三板斧！

## licm

`createLoopInvariantCodeMotionPass`

targetOp 显然是 `LoopLikeOpInterface`，一般把 op 移出 region 的行为叫 `hoist`，感兴趣的同学可以看看 `mlir/lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp`，代码非常有借鉴意义。

# ttir

## add_rewrite_tensor_pointer

`createRewriteTensorPointerPass`：将带有 tensor pointers(由tt.make_tensor_ptr和tt.advance) 的 load、store 以及可能用到 tensor pointer 的 loop 重写为特定的模式。

重写后，方便之后分析 `AxisInfo`。

### 基础信息

- `tt.make_tensor_ptr` 会根据给定的信息和 base ptr 产生 `TensorPointerType`。

```text
// ptr:  !tt.ptr<f16>, shape, stride, offset
// shape, stride 都是 i64 类型的，而 Offset 是 i32 类型的
tt.make_tensor_ptr %ptr, [128, 32], [1, 1], [0, 0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
```

- `tt.advance` 接受 %ptr 和 %offset 两个输入，会对 %ptr 施加 %offse，获得新的 ptr。在 `RegionBranchOpInterface` (尤其是循环)的时候用于返回 `TensorPointerType`，因为 `ptr` 类型本质上受 `basePtr` 和 `offset` 的影响。

```text
scf.for %arg = %c0 to %c32_index step %c1 iter_args(%arg1 = %input) -> (!tt.ptr<tensor<128x32xf16>>) {
  ops
  %advance = tt.advance %arg1, [%c32_i32, %c0_i32] : !tt.ptr<tensor<128x32xf16>>
}
```

- PointerType 的一些形式：
  - !tt.ptr<f32>
  - !tt.ptr<tensor<2xf32>>  这就是 TensorPointerType
  - !tt.ptr<!tt.ptr<f32>>
- tensor<2x3x!tt.ptr<f32>> 这是blockPtr

PointerType 的 `getPointeeType()` 方法会获得 PointerType 内的类型。上面三个分别获得 f32, tensor<2xf32>, !tt.ptr<f32>

- RewritedInfo

用于记录 `tt.make_tensor_ptr` 和 `tt.advance` 产生的信息。

```cpp
Value base; //
SmallVector<Value> shape;
SmallVector<Value> strides;
SmallVector<Value> offsets;
ArrayRef<int64_t> tensorShape; // 即 PointerType 的 PointeeType
```

### 代码逻辑

由于原始 test 文件中的测例太难看了，笔者精简一下，以下面的输入来说明代码逻辑

```text
tt.func public @rewrite_tensor_ptr(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16>>
  %1:2 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0) -> (tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>) {
    %3 = tt.load %arg4 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
    %4 = arith.addf %arg3, %3 : tensor<128x32xf16>
    %5 = tt.advance %arg4, [%c32_i32, %c0_i32] : <tensor<128x32xf16>>
    scf.yield %4, %5 : tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>
  }
  %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  tt.store %2, %1#0 : tensor<128x32x!tt.ptr<f16>>
  tt.return
}
```

1.迭代遍历输入 op (该pass是一个module pass)

```cpp
for (auto &region : op->getRegions()) {
  for (auto &block : region) {
    for (auto &nestedOp : block)
```

笔者也常用这种方法去遍历所有的 op，因为 **walk** 方法不适合迭代遍历。

nits：

- 对于这种接口不会变动且类型确定的类型，可以直接用类名，用 `auto` 感觉代码可读性差一点
- 因为担心break iterator而先收集operation再遍历会多引入空间和时间开销，不妨使用 `llvm::make_early_inc_range(block)`，笔者编译并测试过，不影响功能。
  - [0904update]：我提了pr，已经merge了hhh

```cpp
for (Region &region : op->getRegions()) {
  for (Block &block : region) { // region 的 iterator 不会被影响
    for (Operation &nestedOp : llvm::make_early_inc_range(block)) {
      if (auto newOp = rewriteOp(&nestedOp, eraser)) {
        visitOperation(newOp, eraser);
```

2.对 op 尝试 rewrite，根据 op 类型尝试重写

一般会先处理到产生 `TensorPointerType` 的 op，因为要根据 tt.make_tensor_ptr 和 tt.advance 来记录 `RewritedInfo`，给后面的 userOp 提供转换用的信息。

(1) tt.make_tensor_ptr -> rewriteMakeTensorPtrOp

offset 通过 arith.extsi 扩为 i64

记录 `rewritedInfo`(DenseMap<Value, RewritedInfo>)，其中 key 是 `tt.make_tensor_ptr` 的 result。

```cpp
rewritedInfo[op.getResult()] =
    RewritedInfo(op.getBase(), op.getShape(), op.getStrides(), i64Offsets,
                 tensorType.getShape()); // base, shape, strides, offset,
```

以 `tt.make_tensor_ptr %ptr, [128, 32], [1, 1], [0, 0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>` 为例，得到的 RewritedInfo

```bash
base = %ptr
shapes = {c128_i64, c32_i64}
strides = {c1_i64, c1_i64}
offsets = {c0_i64, c0_i64}
```

(2) scf.for -> rewriteForOp

将 scf.for 上 type 为 `TensorPointerType`(来自`tt.make_tensor_ptr`) 删除，换成当前 `%0` 记录的 `offset` 信息。

```text
%0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16>>
%1:2 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0) -> (tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>) {
->
// %0 和 %1 来自 `!tt.ptr<tensor<128x32xf16>>` 对应的 value 的 rewritedInfo 中的 offset
%0 = arith.extsi %c0_i32 : i32 to i64
%1 = arith.extsi %c0_i32 : i32 to i64
%2:3 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0, %arg5 = %1) -> (tensor<128x32xf16>, i64, i64) {
```

然后将 `rewritedInfo`，将 `%0` 的 `RewritedInfo` 给到 `arg4`，之后 region 内的计算使用的 `TensorPointerType` 就能找到源头。

最后更新 scf.for 的 result 的 `rewritedInfo`。

(3) tt.load / tt.store -> rewriteLoadStoreOp

由于 load 和 store 的处理逻辑大致相同，所以我们以下面 ir 中的 `tt.load` 为例，讲解。

```text
%0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16>>
%1:2 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0) -> (tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>) {
  %3 = tt.load %arg4 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
```

- 保留 `getBoundaryCheck` 信息，得到 `boundaryCheck = array<i32: 1>`

- 根据 `ptr` 中的 `rewritedInfo` 中的信息获得新的 `ptr`。

当前 load 的 `ptr` 对应的 `rewritedInfo` 如下：

```bash
base = %arg0
shapes = {c128_i64, c32_i64}
strides = {c1_i64, c1_i64}
offsets = {c0_i64, c0_i64}
tensorShape = {128, 32}
```

offsets + shapes -> 得到真实的 offsets

```text
%5 = tt.splat %arg4 : i64 -> tensor<128xi64> // offset
%6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32> // shape
%7 = arith.extsi %6 : tensor<128xi32> to tensor<128xi64>
%8 = arith.addi %5, %7 : tensor<128xi64> // offset + stride
%9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
```

乘以对应的 stride 后，再加到 base 上去

```text
%4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>> // base
%10 = tt.splat %c1_i64 : i64 -> tensor<128x1xi64> // stride
%11 = arith.muli %9, %10 : tensor<128x1xi64>
%12 = tt.broadcast %11 : tensor<128x1xi64> -> tensor<128x32xi64>
%13 = tt.addptr %4, %12 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
```

这样循环处理完所有 rank，就得到了最终的 ptr

- 根据 `ptr` 中的 `rewritedInfo` 中的信息以及 `BoundaryCheck`(需要 check 的 rank) 信息，获得新的 `mask`

```text
%c0_i64 = arith.constant 0 : i64
%23 = tt.splat %c0_i64 : i64 -> tensor<1x32xi64> // 下界
%24 = arith.cmpi sge, %18, %23 : tensor<1x32xi64>
%25 = tt.splat %c32_i64 : i64 -> tensor<1x32xi64> // 上界
%26 = arith.cmpi slt, %18, %25 : tensor<1x32xi64>
%27 = arith.andi %24, %26 : tensor<1x32xi1>
%28 = tt.broadcast %27 : tensor<1x32xi1> -> tensor<128x32xi1>
```

- tt.load 根据 `PaddingOption` 去生成 `other` operand

然后根据上述 `ptr`, `mask`, `other` 以及 loadOp 的其他 Attr 信息生成新的 tt.load

(4) tt.advance -> rewriteAdvanceOp

`tt.advance` 接受 %ptr 和 %offset 两个输入，会对 %ptr 施加 %offse，获得新的 ptr。

所以我们只需要 **加上** `%offset` 获得新的 `RewritedInfo`。

```text
%5 = tt.advance %arg4, [%c32_i32, %c0_i32] : <tensor<128x32xf16>>
->
%32 = arith.extsi %c32_i32 : i32 to i64
%33 = arith.addi %arg4, %32 : i64
%34 = arith.extsi %c0_i32 : i32 to i64
%35 = arith.addi %arg5, %34 : i64
```

(5) scf.yield

和 scf.for 中的处理差不多，都是去除 Type 为 `TensorPointerType` 的 operand，换成当前 `RewritedInfo` 中的 `offsets` 信息。

去除上面的例子后，我们还剩下一个 `scf.if -> rewriteIfOp` 没讲。代码看起来其实和 scf.for 的处理逻辑差不多，只不过需要考虑 then region 和 else region。

到现在，该pass所有内容基本记录完，但是小伙伴们可能还有疑问

- 为什么不直接支持 `RegionBranchOpInterface`，而只单独支持了 `scf.for` 和 `scf.if` 呢

笔者猜想应该是 `scf.forall` 是由于在 tensor 上返回值必须为 tensorType(使用tensor.parallel_insert_slice返回)，就不能插入 `i64` 的 blockarg 了，而 `scf.while` 有和 `BranchOPInterface` 中一样的分支跳转，当前也没办法直接给跳转 block 上加 arg。

- 除了 `tt.load` 和 `tt.store` 还有 `tt.atomic_rmw`、`tt.atomic_cas` 能够读写 `ptr`，为什么不支持呢？

这两个 atomic op当前 ptr 类型都不支持 `!ptr<tensor<>>`(TT_PtrLike)。但理论上应该行为和 load 、 store 类似的，我也很疑惑，所以提了个 [issue](https://github.com/triton-lang/triton/issues/4672)。

## add_combine

`createCombineOpsPass`：根据 `td` 中的描述，该 pass 执行的行为是将一些算子改变形式，类似 canoncalizer 中的行为。

使用 `tablegen` 来写几个相似的 `pattern` 的思路真的很不错，见 [Combine.td](https://github.com/triton-lang/triton/blob/main/lib/Dialect/Triton/Transforms/Combine.td)。

- `dot(a, b, 0) + c => dot(a, b, c)` ：当 dot 的 bias 为 0 时，且 dot_res 有且仅有一个 consumer，且为 arith.add，可以合并成新的 dot。

- `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`：将 Offset 信息合并，方便后序的 AxisInfo 分析。

- `broadcast(constant) => reshaped_constant`：当 broadcast 的 operand 是一个 constantOp 时，直接新建一个 resTy 一样大的 constantOp 就好。

剩下的几个 pattern 是在 combine.cpp 中定义的。

- `select(cond, load(ptrs, broadcast(cond), ???), other) => load(ptrs, broadcast(cond), other)`： select 的 condVal 和 tt.load 的 mask 有特殊的关系时，可以合并一下。

其实可以有更多的这类 pattern，例如支持 cond 是 mask 的一个 extract，且 mask 一定是一个 dense value(splat from i1, broadcast from all unit-dims tensor)，所以我尝试提交了一个 [PR](https://github.com/triton-lang/triton/pull/4673)，以支持更多场景。

- `sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1) => dot(x,y,splat(0))`

这个pattern会将特殊情况下的`expand_dim + broadcast + mul + reducesum` 给转为 `tt.dot`，本质上其实是 expand 出的 dim 其实是 reducesum 的 parallel 轴，并不影响 reduction 轴计算结果。（感觉也能改成适用更多case，但我还没想到。）

笔者按照代码流程写了一个输入 ir 帮助理解。

```text
tt.func @test_combine_broadcast_mul_reducesum(%lhs: tensor<128x64xf32>, %rhs: tensor<64x128xf32>) -> (tensor<128x128xf32>) {
    %expand_lhs = tt.expand_dims %lhs {axis = 2 : i32} : tensor<128x64xf32> -> tensor<128x64x1xf32>
    %expand_rhs = tt.expand_dims %rhs {axis = 0 : i32} : tensor<64x128xf32> -> tensor<1x64x128xf32>
    %a = tt.broadcast %expand_lhs : tensor<128x64x1xf32> -> tensor<128x64x128xf32>
    %b = tt.broadcast %expand_rhs : tensor<1x64x128xf32> -> tensor<128x64x128xf32>
    %mul = arith.mulf %a, %b : tensor<128x64x128xf32>
    %reduce = "tt.reduce" (%mul) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 1 : i32} : (tensor<128x64x128xf32>) -> tensor<128x128xf32>
    tt.return %reduce : tensor<128x128xf32>
}
->
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %0 = tt.dot %lhs, %rhs, %cst, inputPrecision = tf32 : tensor<128x64xf32> * tensor<64x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
```

## add_reorder_broadcast

`createReorderBroadcastPass`：根据 td 中的描述，如果 elementwise 的 producer 来自 broadcast 或者 splat，则先进行 elementwise 计算，再进行 broadcast / splat 计算。这样可以减小 elementwise 计算过程中的计算量。

- `elementwise(broadcast(a)) => broadcast(elementwise(a))`
  - 该pattern会考虑 operandVal 来自 splat / dense constVal / broadcast
- `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
  - 该 patten 中也处理了 `elementwise(denseVal(a), denseVal(b), ...) => splat(elementwise(a, b, ...))`， `denseVal` 即 `arith.constant dense<>`

首先，pass会为targetOp掉用 canoncialize 函数来 fold 掉一些op，因为 tt.broadcast 一般配合 tt.expand_dims 来实现维度扩展。

```cpp
template <typename OpType>
class CanonicalizePattern : public OpRewritePattern<OpType> {
public:
  explicit CanonicalizePattern(MLIRContext *context)
      : OpRewritePattern<OpType>(context) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    return OpType::canonicalize(op, rewriter);
  }
};
```

值得注意的是，这个pass中用的是 `OpTraitRewritePattern`，并且将 `match` 函数和 `rewrite` 函数分开了。`matchAndRewrite` 函数在针对特定 operation 或 interface 的 conversion 和 RewritePattern 用得更多吧。

```cpp
struct MoveBroadcastAfterElementwisePattern
    : public OpTraitRewritePattern<OpTrait::Elementwise> {

  MoveBroadcastAfterElementwisePattern(MLIRContext *context)
      : OpTraitRewritePattern(context) {}

  LogicalResult match(Operation *op) const override {
    ...
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    ...
  }
}
```

rewrite 过程中的核心是需要去记录 dense val 最根本的值，对于 splatOp，就是它的 src，对于 dense constant 可以采用 `arith::ConstantOp::materialize` 去捕获。再使用最根本的值去创建新的 elementwise 计算，并以新的 res 创建 tt.splat 去替换使用。

```cpp
DenseElementsAttr constAttr;
if (auto splatOp = llvm::dyn_cast<SplatOp>(definingOp)) {
  scalarOperands[iOp] = splatOp.getSrc();
} else if (matchPattern(definingOp, m_Constant(&constAttr)) &&
           constAttr.isSplat()) {
  auto value = constAttr.getSplatValue<Attribute>();
  scalarOperands[iOp] = arith::ConstantOp::materialize(
      rewriter, value, constAttr.getElementType(), loc);
}
```

## add_loop_unroll

[0914update]：突然发现又在 ttir 层加了一个`createLoopUnrollPass`，补充一下。

这个pass会匹配 `scf.for` op 上名为 `tt.loop_unroll_factor` 的 Attribute，这是个 `IntegerAttr`，默认设为1，作为 `unrollFactor` 。然后根据这个值去调用 mlir 官方的 [loopUnrollByFactor](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/Utils/Utils.cpp#L375)。

pass 结束后， for 循环会按照这个值展开，展开的行为是每`unrollFactor`次相邻的计算展开。 比如

```text
scf.for 0(lb) to 10(ub) step 1, (tt.loop_unroll_factor = 3)
  ops
->

scf.for 0(lb) to 9(ub) step 3 // 每三个相邻的循环合并
  ops
  ops
  ops
scf.for 9(lb) to 10(ub) step 1 // 防止还有剩下的循环
```

以官方提供的例子来看，我们以 lb = 0, ub = 10(即%arg1), step = 2 为例

```text
// 小改一下，让func有返回值，这样可以跑canonicalize pass来fold掉一些arith计算。然后把 step 设为 2
// triton-opt  -triton-loop-unroll test/Triton/loop-unroll.mlir --canonicalize
tt.func @add_kernel_unroll(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) -> tensor<256xf32> {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // lb = 1, ub = 10, step = 2 (即1、3、5、7、9)
  %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
      %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %4 = arith.addf %arg4, %3 : tensor<256xf32>
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  } {tt.loop_unroll_factor = 3 : i32}
  tt.return %2#0 : tensor<256xf32>
}
```

unroll pass 结束后：

```text
//这里就简单记录下pass后的ir
  %0 = arith.divui %arg1, %c2_i32 : i32 // 10 / 2 = 5
  %1 = arith.remsi %0, %c3_i32 : i32    // 5 % 3 = 2
  %2 = arith.subi %0, %1 : i32          // 5 - 2 = 3
  %3 = arith.muli %2, %c2_i32 : i32     // 3 * 2 = 6
  // lb = 1, ub = 7, step = 6 (即1、3、5次循环合并)
  %5:2 = scf.for %arg2 = %c1_i32 to %4 step %c6_i32 iter_args(%arg3 = %cst_0, %arg4 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
  ...
  // lb = 7, ub = 10, step = 2 (即7、9)
  %6:2 = scf.for %arg2 = %4 to %arg1 step %c2_i32 iter_args(%arg3 = %5#0, %arg4 = %5#1) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
```

unroll pass 对 scf.for 做了什么就到这了，让我们回到 mlir 官方的 [loopUnrollByFactor](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/Utils/Utils.cpp#L375)，简单介绍一下。

我们发现这段代码中其实将静态和动态的情况分开处理了，因为静态的情况下能分析出需不需要第二个循环来处理没有被处理完的数据。处理的流程也是一连串的 arith.op 来计算新的 for 中的 lb, ub, step，这就不展开讲。有一个用处很多的函数 `getConstantIntValue` 用来获得 `OpFoldResult` 中的真实值，这个挺好用的。

```cpp
// mlir/lib/Dialect/Utils/StaticValueUtils.cpp
std::optional<int64_t> getConstantIntValue(OpFoldResult ofr) {
  // Case 1: Check for Constant integer.
  if (auto val = llvm::dyn_cast_if_present<Value>(ofr)) {
    APSInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal.getSExtValue();
    return std::nullopt;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = llvm::dyn_cast_if_present<Attribute>(ofr);
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue();
  return std::nullopt;
}
```

思考：

1.attr 的名称

代码中写做下面的内容，用 `static const` 来修饰一个字符串，作为编译期的 hint。

```cpp
static const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
```

在 mlir 中更常见的关于使用 string 作为编译 hint 的写法是 `static constexpr StringLiteral`(当然这无关紧要)

```cpp
static constexpr StringLiteral loopUnrollFactorAttrName("tt.loop_unroll_factor");
```

至于这么写有什么优点，我目前只了解使用字面量更节省内存，相比 `const` 强调不应该被修改， `constexpr` 更强调编译期不可变。

更多这样的小细节欢迎收看我的 [mlir编程笔记](https://tfruan2000.github.io/posts/mlir-code-note/#llvm)。

2.谁来给 `scf.for` 主动设置这个 attr

`scf.for` 直接由 triton-lang 中的 for 循环下降而来。

截止0914 triton仓库中的代码，我没有看到这个 `tt.loop_unroll_factor` attr 在下降流程中谁主动给挂到 op 上，笔者猜想应该是在写 triton-lang 的时候程序员直接加到 for 循环上的。

3.为什么不支持affine.for

在 mlir 中， affine.for 也支持了 unroll pattern，但目前在 triton 中并不会下降出 affine op(没有场景)，所以当前该pass的锚点是 `scf.for`。
