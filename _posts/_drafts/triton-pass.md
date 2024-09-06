# 从一个 MLIR Programer 的角度来读 OpenAI Triton 源码

# 前言

OpenAI [Triton](https://github.com/triton-lang/triton/tree/main) 是什么？这个问题很多大佬都已经回答过了，阅读完以下 blog 相信大家会有个基础的理解。

- 杨军老师的 [谈谈对OpenAI Triton的一些理解](https://zhuanlan.zhihu.com/p/613244988)，帮助大家建立一个宏观印象
- 董鑫大佬的 [如何入门 OpenAI Triton 编程?](https://www.zhihu.com/question/622685131/answer/3217107882)，帮助了解更多关于语法上的信息
- BobHuang大佬的 [浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)，帮助初学者大致明白一段 python 程序如何进入 triton pipeline，然后跑起来。

有一些基础的认识后，大家就可以在 Triton 中各取所需了。作为一个 MLIR Programer，我还希望了解每个 transform pass 和 每一步 conversion 做了什么事情，所以本文是个人读源码的一个记录(枯燥预警!!)。

> 当前 Triton 中关于 arch 和 non-arch 的抽象界限比较模糊，所以 transform 和 conversion 中可能含有很多 hardware-special 的信息，所以文中会夹带我对 arch 的理解作为源码解读的补充，若有不对，烦请指正。
{: .prompt-info }

下文皆以 NVGPU 为 target 去说明 triton 的相关信息。本文基于 triton 版本：[6083043](https://github.com/triton-lang/triton/tree/6083043eb7a0722db6bd2ad8efc453300da74819)。**本文并不严格follow triton ir 下降的步骤。** 后续版本可能发生了 pass 改名、增加/删除、实现修改，请自行核对。

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

`createRewriteTensorPointerPass`

将带有 tensor pointers(由tt.make_tensor_ptr和tt.advance) 的 load/store 重写为特定的模式。

`tt.make_tensor_ptr` 会根据给定的信息和 base ptr 产生 `TensorPointerType`。

```text
// ptr:  !tt.ptr<f16>, shape, stride, offset
tt.make_tensor_ptr %ptr, [128, 32], [32, 1], [0, 0] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
```

`tt.advance` 接受 %ptr 和 %offset 两个输入，会对 %ptr 施加 %offse，获得新的 ptr。

> PointerType 的一些形式：
>
> - !tt.ptr<f32>
> - !tt.ptr<tensor<2xf32>>  这就是 TensorPointerType
> - !tt.ptr<!tt.ptr<f32>>
>
> PointerType 的 `getPointeeType()` 方法会获得 PointerType 内的类型。上面三个分别获得 f32, tensor<2xf32>, !tt.ptr<f32>
{: .prompt-info }

## add_combine

`createCombineOpsPass`

## add_reorder_broadcast

`createReorderBroadcastPass`

## add_convert_to_ttgpuir

`createConvertTritonToTritonGPUPass`

