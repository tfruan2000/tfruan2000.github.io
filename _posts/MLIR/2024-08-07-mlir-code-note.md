---
title: MLIR Code Note
author: tfruan
date: 2024-08-07 20:00:00 +0800
categories: [MLIR]
tags: [MLIR, Code Note]
---

> 这只是平时写代码时随手记录的一些笔记，方便自己回顾如何写mlir的代码，一般都在我的github上更新 [github-mlir-note](https://github.com/tfruan2000/mlsys-study-note/blob/main/ai_compiler/MLIR/MLIR_Note.md)
{: .prompt-info }


# 工欲善其事，必先利其器

## 推荐项目

- 学习 mlir 的项目

[mlir-tutorial](https://github.com/j2kun/mlir-tutorial) 使用 `bazel` 构建项目，相比 `cmake` 构建个人感觉更适合新手。

- 可以抄代码的项目

[IREE](https://github.com/iree-org/iree) 架构、风格上很有 Google 的风范。

[ByteIR](https://github.com/bytedance/byteir) ，字节开源项目。

## 跳转工具clangd

`vscode` 专属。

1.首先我们需要生成 `compile_commands.json`，以编译 `llvm` 为例：

- 如果是cmake编译

```bash
# mac上编译mlir
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_BUILD_EXAMPLES=ON

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

生成的 `compile_commands.json` 在 `build` 目录下，复制到llvm-project目录即可

- 如果是bazel编译

在BUILD文件配置一下下面的内容，再bazel run 一下就可以编译出compile_commands.json
详情自学：[https://github.com/hedronvision/bazel-compile-commands-extractor/tree/main](https://github.com/hedronvision/bazel-compile-commands-extractor/tree/main)

(1) 修改WORKSPACE，添加

```bash
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "hedron_compile_commands",

    # 记得把下面两处 commit hash 换成 github 上最新的版本
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/ed994039a951b736091776d677f324b3903ef939.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-ed994039a951b736091776d677f324b3903ef939",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()

```

(2) 在根目录下的 `BUILD.bazel` 中添加下面语句

```bash
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    # 指定目标 target 及其编译选项/参数，例如 `mlir-opt` 、`config=clangd`
    targets = {
      "//:my_output_1": "--important_flag1 --important_flag2=true"
    },
)
```

(3) 运行 `bazel run :refresh_compile_commands`

2.然后，配置vscode的clangd插件

`ctrl + p` 输入 clangd，先点击 下载language server；然后 加 settings.json , `ctrl + p` 打开工作区设置json，将以下内入加入

```cpp
{
    "clangd.arguments": [
        "--header-insertion=never",
        "--compile-commands-dir=${workspaceFolder}/",
        "--query-driver=**",
    ]
}
```

使用compile_commands.json主要是方便索引文件，特别是td生成的 `inc` 文件，但也可以人为从 `build/tools/mlir/include/mlir/xxx/xxx` 中找到编译出的 `inc`。

## 代码格式

一般使用 `clang-format` 工具(或者基于此的 `lint.sh`)。

安装
```bash
apt-get install clang-format
```

创建`.clang-format`
```bash
BasedOnStyle: LLVM
ColumnLimit: 80
```

格式化
```bash
# 单个文件
clang-format -i path/to/your/file.cpp
# 整个目录
find path/to/your/project -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

---

# Adaptor

只有**operands没有results**的中间态，可以从adaptor中获得很多基础信息

`ConversionPattern` 相比 `RewriterPattern` 需要多传递一个 `adaptor`

1.OpAdaptor的作用：封装了op的operands

2.ConversionPattern和RewritePatter的区别

- ConversionPattern常配合 **applyFullConversion/applyPartialConversion** 使用，用于dialect2dialect的op之间变换

- RewritePattern一般用于优化变换，常配合 **applyPatternAndFoldGreedily** 使用

```cpp
// OpConversionPattern
struct AbsOpToMathAbsConverter : public OpConversionPattern<mhlo::AbsOp> {
  using OpConversionPattern<mhlo::AbsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mhlo::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
...

// OpRewritePattern
struct TransposeSliceLayoutPattern : public OpRewritePattern<mhlo::SliceOp> {
  using OpRewritePattern<mhlo::SliceOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(mhlo::SliceOp op,
                  OpRewritePattern &rewriter) const override {
```

---

# Analysis

## Analisys Manager

```bash
mlir/include/mlir/Pass/AnalysisManager.h
```
`Analyses` 是独立于其他设施的数据结构，可以将相关的信息 perserve 起来。

例如 `Transforms/CSE.cpp` 中就将一些 `Analyses` 信息保存给下一次分析。
```cpp
  // If there was no change to the IR, we mark all analyses as preserved.
  if (!changed)
    return markAllAnalysesPreserved();

  // We currently don't remove region operations, so mark dominance as
  // preserved.
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
```

但使用 `markAnalysesPreserved` 在 `pass` 间传递信息的行为是不可取的，因为该功能只是为了减少编译时间，要在 `pass` 间传递信息最合理的方法是设计一套 `Attribute` 挂在 op上。

## Dataflow Framework

```bash
mlir/include/mlir/Analysis/DataFlowFramework.h
mlir/lib/Analysis/DataFlowFramework.cpp
```

1.ChangeResult

```cpp
enum class [[nodiscard]] ChangeResult {
  NoChange,
  Change,
};
```

> `[[nodiscard]]` 来标记函数的返回值不应该被忽略。也就是说，当调用一个被标记为 `[[nodiscard]]` 的函数时，
> 如果返回值没有被使用，编译器会发出警告。

2.ProgramPoint

ProgramPoint 是一个 `PointerUnion`，可以是 `Operation *, Value, Block *`

3.DataFlowSolver

实现 child data-flow analyses，使用的是 fixed-point iteration 算法。
一直维护 `AnalysisState` 和 `ProgramPoint` 信息。

数据流分析的流程：

(1) 加载并初始化 children analyses
例如
```cpp
std::unique_ptr<mlir::DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<mlir::DataFlowSolver>();
  solver->load<mlir::dataflow::DeadCodeAnalysis>();
  solver->load<mlir::dataflow::SparseConstantPropagation>();
  ...
  return solver;
}
```

(2) 配置并运行分析，直到达到设置的 fix-point

```cpp
if (failed(solver->initializeAndRun(root))) {
  LLVM_DEBUG(llvm::dbgs() << " - XXX analysis failed.\n");
  return failure();
}
```

(3) 从 solver 中 query analysis state results

```cpp
// lookupState 可能返回 null
const auto analysisState = solver->lookupState<xxxxx>(op)
```

## Liveness

```bash
mlir/include/mlir/Analysis/Liveness.h
mlir/bin/Analysis/Liveness.cpp
```

- 对 op ->  Liveness(Operation *op)

- 对 block -> liveness.getLiveness(block) -> LivenessBlockInfo

## AliasAnalysis

## LocalAliasAnalysis

```bash
mlir/include/mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h
mlir/lib/Analysis/AliasAnalysis/LocalAliasAnalysis.h
```

1.AliasResult: 两个location之间是否有关系
- Kind
  - NoAlias
  - MayAlias
  - PartialAlias : 两个loc互相alias，但是部分重叠
  - MustAlias
- isNO / isMay / isPartial / isMust -> bool

2.AliasResult alias(Value lhs, Value rhs);

```cpp
// 确定一个op是否对一个value有读/写行为
bool isOpReadOrWriteInplace(Operation *op, Value val) {
  auto memInterface = llvm::dyn_cast<MemoryEffectOpInterface>(op);
  if (!memInterface)
    return false;
  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  memInterface.getEffects(effects);
  bool readOnVal = false;
  bool writeOnVal = false;
  LocalAliasAnalysis analysis;
  for (MemoryEffects::EffectInstance effect : effects) {
    if (llvm::isa<MemoryEffects::Read>(effect.getEffect()) &&
        !analysis.alias(val, effect.getValue()).isNo()) {
        readOnVal = true;
    }
    if (llvm::isa<MemoryEffects::Read>(effect.getEffect() &&
        !analysis.alias(val, effetc.getValue()).isNo()) {
      writeOnVal = true;
    }
  }
  return readOnVal || writeOnVal;
}
```

3.collectUnderlyingAddressValues

重载了多种形式，常用的有以下输入：

- (Value, SmallVectorImpl<Value> &output)

- (OpResult result, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)
  - result.getOwner() -> ViewLikeOpInterface -> 继续调用 viewOp.getViewSource()
  - result.getOwner() -> RegionBranchOpInterface

- (BlockArguement arg, unsigned maxDepth, DenseSet<Value> &visited, SmallVectorImpl<Value> &output)

## SliceAnalysis

用来遍历 use-def 链的 analysis
一般可以将 use-def 理解为
- def : 写
- use : 读

```bash
 ____
 \  /  defs (in some topological order)
  \/
  op
  /\
 /  \  uses (in some topological order)
/____\
```

1.getForwardSlice : 获得root op的use链 (向ir的结尾找)

```bash
从 0 开始追，可以获得 {9, 7, 8, 5, 1, 2, 6, 3, 4}

              0
   ___________|___________
   1       2      3      4
   |_______|      |______|
   |   |             |
   |   5             6
   |___|_____________|
     |               |
     7               8
     |_______________|
             |
             9
```

输入， root可以是op，也可以是value
```cpp
void getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});

void getForwardSlice(Value root, SetVector<Operation *> *forwardSlice,
                     const ForwardSliceOptions &options = {});
```

2.getBackWardSlice : 获得root op的def链 (向ir的开头找)

```bash
从 node 8 开始， 可以获得 {1, 2, 5, 3, 4, 6}

   1       2      3      4
   |_______|      |______|
   |   |             |
   |   5             6
   |___|_____________|
     |               |
     7               8
     |_______________|
             |
             9
```

输入， root可以是op，也可以是value

```cpp
void getBackwardSlice(Operation *op, SetVector<Operation *> *bac
                      const BackwardSliceOptions &options = {});

void getBackwardSlice(Value root, SetVector<Operation *> *backwa
                      const BackwardSliceOptions &options = {});
```

3.SliceOptions
- TransitiveFilter filter : 设置遍历条件，当遍历到的节点不符合 filter 时停止(注意第一个遍历对象就是 rootOp)

- bool inclusive : 返回的 sliceSetVec中 是否包含 rootOp

**ForwardSliceOptions** : using ForwardSliceOptions = SliceOptions;

**BackwardSliceOptions** : 相比 SliceOptions 多一个参数 ` bool omitBlockArguments`，这个参数控制是否避免遍历 blockArguement

```cpp
BackwardSliceOptions sliceOptions;
// 不遍历 blockArg(可以理解为到blockArg这就结束)
sliceOptions.omitBlockArguments = true;

// 所有加入backwardSlice的op都需要满足以下条件
// 第一下会遍历本身
sliceOptions.filter = [rootOp](Operation *slice) -> bool {
  return !llvm::isa<arith::ConstantOp, tensor::EmptyOp, memref::AllocOp，
                    scf::ForallOp, scf::ForOp, scf::IfOp>(slice)
          && rootOp->isProperAncestor(slice);
};

SmallVector<Operation *> backwardSlice;
getBackwardSlice(targetOp, &backwardSlice, sliceOptions);
```

---

# Attribute

```cpp
mlir/include/mlir/IR/Attribute.h
```

常见类型：

- StringAttr
- UnitAttr / IntegerAttr / IndexAttr
- BoolAttr
- ArrayAttr
- DictionaryAttr

常用方法：

1.使用 `OpBuilder` 可以创建这类 `Attr`，例如
rewriter.getI64IntegerAttr 或builder.getI64IntegerAttr。

2.src: AttrTy
- get() 例如从SmallVector<Attribute>变成ArrayAttr
  ```cpp
  SmallVector<Attribute, 8> mappings;
  ArrayAttr tmp = ArrayAttr::get(context, mappings)
  ```

- getName()

- setValue()

- getValue() 对于IntegertAttr会返回APInt，之后一般可以接 `getSExtValue()` ，来将APInt转为**int64_t**

- src : operation*
  - getAttr / getAttrOfType ，一般get完之后要cast到对应的AttrType，例如
    ```cpp
    op->getAttr(getAliasAttrKey()).dyn_cast_or_null<mlir::IntegerAttr>()
    op->getAttrOfType<DenseI64ArrayAttr>
    ```

  - hasAttr / hasAttrOfType

  - setAttr(StringRef name, Attribute value)
    - name可以`constexpr llvm::StringLiteral` 形式定义在头文件中
    - funcOp→setAttr(attrName, IntegerAttr::get(intType, 1));

  - removeAttr

  - func::FuncOp::setResultAttr

## operation、attribute、type关系

| 专用指针      | 通用指针     | 值               |
|-------------|-------------|------------------|
| AddOp       | Operation * | Operation        |
| IntegerType | Type        | TypeStorage      |
| IntegerAttr | Attribute   | AttributeStorage |

---

# Block

```cpp
mlir/include/mlir/IR/Block.h
```

Block 包含 `BlockArgument`（使用getArguements()获得）和 `BlockOperand`

## BlockArgument

继承自 `Value`。

`Block *getOwner()` 返回该arg属于哪个block。

`unsigned getArgNumber()` 返回该arg的index。

## BlockOperand

继承自`IROperand`。

`unsigned getOperandNumber()` 返回该operand的index。

## 使用

1.返回 Block

- `Operation *` -> `getBlock()`
- `Value` -> `getParentBlock()`


2.遍历block
- walk
  ```cpp
  block->walk([&](Operation *op) {...
  ```

- 只遍历同层op
  ```cpp
  Operation &workOp : rootBlock->getOperations()
  ```

---

# Builder

# Builder

```cpp
mlir/include/mlir/IR/Builders.h
mlir/lib/IR/Builders.cpp
```

`Builder` 用于创建新的 MLIR 操作，例如各种 `Type`, `Attr`, `Affine Expressions` 等

## OpBuilder

OpBuilder 继承自 Builder 类，**额外提供了struct Listener和class InsertPoint**

1.InsertPoint

```cpp
Listener *getListener() const { return listener; }
void clearInsertionPoint();
InsertPoint saveInsertionPoint();

// insertionPoint设在block内的iterator处
void setInsertionPoint(Block *block, Block::iterator insertPoint);

// insertionPoint设到op前面，本质上还是找到op在block内的iterator
void setInsertionPoint(Operation *op) {
  setInsertPointPoint(op->getBlock(), Block::iterator(op));
}

// insertionPoint设到op后面
void setInsertionPointAfter(Operation *op) {
  setInsertPointPoint(op->getBlock(), ++Block::iterator(op));
}

// insertionPoint设到value后面
void setInsertionPointAfterValue(Value val) {
  if (Opeartion *op = val.getDefiningOp()) {
    setInsertionPointAfter(op);
  } else {
    auto blockArg = llvm::cast<BlockArguement>(val);
    setInsertionPointToStart(blockArg.getOwner());
  }
}

// insertionPoint设到block开头
void setInsertionPointToStart(Block *block);

// insertionPoint设到block结尾
void setInsertionPointToEnd(Block *block);
```

2.create

```cpp
Block *createBlock(Region *parent, Region::iterator insertPt = {},
                   TypeRange argTypes = std::nullopt,
                   ArrayRef<Location> locs = std::nullopt);
// createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
Operation *insert(Operation *op);
Operation *create(const OperationState &state);
```

- OpTy create(loc, Args &&..args);
先创建  `OperationState` 对象，再调用 `OpTy::build` 方法创建 `Operation` 对象
- createOrFold
返回值是 Value （也可以直接作为 OpFoldResult 使用)
创建op后立即尝试fold，一般在创建某些有xxxOp.cpp中有opFoldPattern的op时使用，例如一些arith dialect 中的op 以及 memref.dim
> 参见: mlir/lib/Dialect/Complex/IR/ComplexOps.cpp

3.clone

```cpp
Operation *clone(Operation &op, IRMapping &mapper);
Operation *clone(Operation &op);
Operation *cloneWithoutRegions(Operation &op, IRMapping &mapper) {
  return insert(op.cloneWithoutRegions(mapper));
}
Operation *cloneWithoutRegions(Operation &op) {
  return insert(op.cloneWithoutRegions());
}
```

例：使用linalg.reduce的region创建一个linalg.map

```cpp
// op 是 linalg.reduce
Value emptyOp = rewriter.create<tensor::EmptyOp>(
    loc, initDims, dstType.getElementType());
auto mapOp = rewriter.create<linalg::MapOp>(
    loc, ValueRange(op.getDpsInputs()), emptyOp,
    [&](OpBuilder &b, Location loc, ValueRange args) {});

// 下面的代码等价于 rewriter.inlineRegionBefore(op->getRegion(0), mapOp->getRegion(0), mapOp->getRegion(0)->begion());

Block *opBody = op.getBody();
llvm::SmallVector<Value> bbArgs;
for(Operation *opOperand : op.getOpOperandsMatchingBBargs()) {
  bbArgs.emplace_back(opBody->getArgument(
      opOperand->getOperandNumber()));
}
Block *mapOpBody = mapOp.getBlock();
SmallVector<BlockArgument> mapOpBbargs;
for (OpOperand *opOperand : mapOp.getOpOperandsMatchingBBargs()) {
  mapOpBbargs.emplace_back(mapOpBody->getArgument(opOperand->getOperandNumber());
}
assert(mapOpBbargs.size() == bbArgs.size());
IRMapping bvm;
for (auto [bbarg, newBBarg] : llvm::zip(bbArgs, mapOpBbargs)) {
  bvm.map(bbarg, newBBarg);
}
rewriter.setInsertionPointToStart(mapOpBody);
for (Operation &operation : *reduceOpBody) {
  rewriter.clone(operation, bvm);
}
```

## Listener

Listener用于hook到OpBuilder的操作，Listener继承自 ListenerBase，ListenerBase有两种 kind

```cpp
// Listener() : ListenerBase(ListenerBase::Kind::OpBuilderListener)
struct ListenerBase {
  enum class Kind {
    OpBuilderListener = 0,
    RewriterBaseListener = 1
  };
  ...
}
```

Listener常用两个函数为 `notifyOperationInserted(Operation *Op)` 和 `notifyBlockCreated(Block *block)`。自定义rewriter时，一般需要 `override` 这两个函数。

`ForwardingListener` 可以将所有 `notify` 发送给另外一个 `OpBuilder::Listener`，用于创建监听链条

```cpp
struct ForwardingListener : public RewriterBase::Listener {
  ForwardingListener(OpBuilder::Listener *listener) : listener(listener) {}
```

## RewriterBase

```cpp
mlir/include/mlir/IR/PatternMatch.h
mlir/lib/IR/PatternMatch.cpp
```

继承自 OpBuilder，且将 Listener 设置为 RewriterBaseListener

```cpp
class RewriterBase : public OpBuilder {
public:
  struct Listener : public OpBuilder::Listener {
    Listener() : OpBuilder::Listener(Kind::RewriterBaseListener) {}
  };
}
```

常用函数：

1.notify ： 在正式对op修改前都需要调用notify，以便listener监听

- notifyOperationModified : in-place 修改

- notifyOperationReplaced : 调用 replaceOp时触发
  ```cpp
  if (auto *listener = dyn_cast_if_present<RewriteBase::Listener>(rewriter.getListener())) {
    listener->notifyOperationReplaced(op, existing);
  }
  rewriter.replaceAllUsesWith(op->getResults())
  opsToErase.push_back(op);
  ```
- notifyOperationErased : 调用 earseOp时触发

2.modifyOpInPlace : 会调用 `startOpModification` 和 `finalizeOpModification`

```cpp
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
```

3.replaceAllUsesWith

## IRRewriter

继承自 `RewriteBase`，当 `PatternRewriter` 不可用时才使用

```cpp
class IRRewriter : public RewriterBase {
public:
  explicit IRRewriter(MLIRContext *ctx, OpBuilder::Listener *listener = nullptr)
      : RewriterBase(ctx, listener) {}
  explicit IRRewriter(const OpBuilder &builder) : RewriterBase(builder) {}
};
```

## PatternMatch

1.PatternBenefit

一般配合 `Pattern` 使用，表示一个pattern的benefit，benefit越高越先apply

```cpp
patterns.add<DoWhileLowering>(patterns.getContext(), /*benefit=*/2);
```

benefit的取值范围为 **0到65535**

2.Pattern

```cpp
class Pattern {
  /// This enum represents the kind of value used to select the root operations
  /// that match this pattern.
  enum class RootKind {
    Any,
    OperationName,
    InterfaceID,
    TraitID
  };
 ...
```

有match、rewrite、matchAndRewrite这些函数，也会设置 `PatternBenefit` (默认为1)

3.RewritePattern

继承自pattern

```cpp
  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    if (succeeded(match(op))) {
      rewrite(op, rewriter);
      return success();
    }
    return failure();
  }
```

一些子类：

(1)OpOrInterfaceRewritePatternBase

- **OpRewritePattern** : 使用 SourceOp::getOperationName() 来match

- **OpInterfaceRewritePattern** : 使用 SourceOp::getInterfaceID() 来match
```cpp
struct AddOpPat : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter & rewriter) const override{

static EraseDeadLinalgOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(LinalgOp op,
                                PatternRewriter &rewriter) const override{
```

(2)OpTraitRewritePattern

-  使用 TypeID::get<TraitType>() 来match

- 例如某些elementwiseTrait : `OpTraitRewritePattern<OpTrait::Elementwise>`

4.RewritePatternSet

```cpp
RewritePatternSet(MLIRContext *context,
                  std::unique_ptr<RewritePattern> pattern)
    : context(context) {
  nativePatterns.emplace_back(std::move(pattern));
}
```

(1)新建pattern

所以一般新建 `RewritePatternSet` 对象时都得传入 context

```cpp
RewritePatternSet patterns(&getContext());
```

然后再一些函数来归类pattern

```cpp
populateAffineToStdConversionPatterns(patterns);
void mlir::populateAffineToStdConversionPatterns(RewritePatternSet &patterns) {
    ...
}
```

也可以通过PDLL来写pattern(包含constrict和rewrite)
```cpp
RewritePatternSet(PDLPatternModule &&pattern)
    : context(pattern.getContext()), pdlPatterns(std::move(pattern)) {}
```

(2)add : 向set中添加pattern

```cpp
add(LogicalResult (*implFn)(OpType, PatternRewriter &rewriter),
    PatternBenefit benefit = 1, ArrayRef<StringRef> generatedNames = {})
```

(3)clear : 清空set中的pattern

5.PatternRewriter

继承自 `RewriteBase`， 用于重写（transform）现有 MLIR 操作的工具。它提供了一组方法，允许用户在遍历操作并修改它们时进行规则匹配和替换。在rewrite pattern中才使用

- `PatternRewriter &rewriter`

- `ConversionPatternRewriter &rewriter` : 相比pattern rewriter要多传入一个adaptor，详细见 Conversion 节

常用操作

(1)设置插入点（与builder同）
- setInsertionPoint(Operantion *)
- setInsertionPointAfter

(2)block

getBlock()

(3)创建
- create<OpTy>(…)
- create(OperationState)
  ```cpp
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, op->getAttrs(), op->getSuccessors());
  Operation *newOp = rewriter.create(state);
  ```

(4)替换
- replaceOp(Operation *op, Operation *newOp)

- replaceOp(Operation *op, ValueRange newValues())

例如getResults()作为ValueRange输入

- replaceAllOpUsesWith(Operation \*from, ValueRange to) / replaceAllOpUsesWith(Opeation \*from, Operation \*to )

- replaceUsesWithIf(Value from, Value to, func_ref) / replaceUsesWithIf(ValueRange from, Value to, func_ref) / replaceUsesWithIf(Operation \*op, Value to, func_ref)
  ```cpp
  // 替换forallOp外的使用
  rewriter.replaceAllUsesWithIf(workOp->getResult(0), forallOp->getResults(idx)
    [&](OpOperand use) {return !forallOp->isProperAncestor(use.getOwner())
  // 仅替换当前op的使用
  rewriter.replaceUsesWithIf(emptyOp->getResult(), newEmptyOp->getResult(),
      [&](OpOperand use) { return use.getOwner() == op; });
  ```

- replaceAllUsesExcept(Value from, Value to, Operation *exceptedUser) 本质是使用 `replaceUsesWithIf` 来实现
  ```cpp
  rewriter.replaceUsesWithIf(from, to,
      [&](OpOperand use) { return use.getOwner() != exceptedUser; });
  ```

(5)消除
- earseOp(Operation *op) : 如果要在pattern中删除op，最好使用 `rewriter.earseOp`，使用op自带的 `erase` 函数代码运行时会在debug模式出问题

- earseBlock(Block *block)

示例

```cpp
struct AddOpPat : public OpRewritePattern<AddOp> {
	using OpRewritePattern<AddOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(AddOp op,
		PatternRewriter & rewriter) const override{
	xxx
	return success();
}
};

class AddOpPatPass : public impl::AddOpPatPassBase<AddOpPatPass> {
	explicit AddOpPatPass() = default;
	void runOnOperation() override {
		RewriterPatternset patterns(&getContext());
		patterns.add<AddOpPat>(patterns.getContext());
		if (failed(applyPatternAndFlodGreedily(getoperation(), std::move(patterns))))
			return signalPassFailure();
	};
}

std::unique_ptr<pass> mlir::createAddOpPatPass() {
	return std::make_unique<AddOpPatPass>;
}
```


---

# Bufferize

## bufferization dialect

bufferization：将逻辑计算语义的tensor转为物理内存语义的buffer

- bufferize::AllocTensorOp

申请一块空间，使用给定shape创建一个bufferize allocation。常会传入一个可选的 srcOp，表示从这个srcOp拷贝出的数据，此时就传入的 ValueRange dynamicShape就应为空。

该op主要是帮助bufferization过程提供一个 `handler`，并且这样产生的alloc_tensor op没有不会产生 read-after-write 冲突，也不会alias其他buffer，可以再进行 `in-place bufferize`

## one-shot-bufferize

（copy from大佬）

```cpp
mlir/lib/Dialect/Bufferization/IR/BufferizableOpInterface.cpp
```

1.OneShotBufferize pass

对于每个有 `BufferizableOpInterface` 的op都进行bufferize

- 声明：mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td
    - 1) 先基于tensor的SSA use-def链进行原位分析来确认哪些operand可以**in-place bufferize**.（尽量减少alloc和copy, 提高性能）
        - destination-passing style op（继承`DestinationStyleOpInterface` ）： 某一个**operand和dst的buffer可复用**，所以分配了该operand的buffer后，无需再分配dst的buffer：eg: %t0 = tensor.insert %f into %dest[%idx]， buffer(%t0)和buffer(%dest)是完全一致的；
        - 非destination-passing style op：对每个OpOperand产生一个新的buffer allocation, eg：tensor.generate
        - 所有新allocate的buffer后续都会deallocate，不然会内存泄露
    - 2) TensorCopyInsertion：对确定是**out-of-place的operands插入 copies**，insertTensorCopies()函数。
    - 3) 调用bufferize接口bufferize()函数来实现bufferize. bufferizeOp()函数。
    - 4) 函数签名的layout map由`function-boundary-type-conversion`选项单独控制，可选的参数有3种：`infer-layout-map`，`fully-dynamic-layout-map` and `identity-layout-map`， 默认是`infer-layout-map`。无法精确推测时，函数参数类型为fully dynamic layout maps。
    - 5)  `bufferize-function-boundaries` 是一个用来对funcOp、returnOp、callOp进行bufferize的flag
    - 6) funcArg一般可以bufferize，除非有 `bufferization.writable = false`
- 实现：mlir/lib/Dialect/Bufferization/Transforms/Bufferize.cpp
    - struct OneShotBufferizePass {void runOnOperation() override }
        - Configure type converter， 先获得 unknownTypeConversionOption：
            - 若是LayoutMapOption::IdentityLayoutMap， bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace)；
            - 否则，只能是LayoutMapOption::FullyDynamicLayoutMap，bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,memorySpace);
        - Configure op filter. 依据编译选项设置不可bufferize的op
        - 依据编译选项是否激活bufferizeFunctionBoundaries确定调用哪个函数进行bufferize:
            - 若激活了，runOneShotModuleBufferize(moduleOp, opt, &statistics)
            - 反之，runOneShotBufferize(moduleOp, opt, &statistics)
        - createCanonicalizerPass()
        - createCSEPass()
        - createLoopInvariantCodeMotionPass()
- 示例：mlir/test/Dialect/Bufferization/Transforms/one-shot-module-bufferize-out-params.mlir, mlir/test/Dialect/Bufferization/Transforms/one-shot-module-bufferize.mlir

2.transform IR : transform.bufferization.one_shot_bufferize 有很多可选的参数

- layout{IdentityLayoutMap} { bufferize_function_boundaries = true }
- {bufferize_function_boundaries = true }
- 定义：mlir/include/mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.td
- 实现：transform.bufferization.one_shot_bufferize的代码：
    - mlir/lib/Dialect/Bufferization/TransformOps/BufferizationTransformOps.cpp: transform::OneShotBufferizeOp::apply()函数，从transform IR提供的各个参数中获得OneShotBufferizationOptions options，之后主要调用
        - runOneShotModuleBufferize()
            - insertTensorCopies(moduleOp, options)
            - bufferizeOp() 会调用`BufferizableOpInterface::bufferize()`函数来对每个op进行具体的bufferize
        - runOneShotBufferize()
            - insertTensorCopies(target, options)
            - bufferizeOp() 会调用`BufferizableOpInterface::bufferize()`函数来对每个op进行具体的bufferize
- 示例：mlir/test/Dialect/Bufferization/Transforms/transform-ops.mlir

```mlir
// 编译命令：mlir-opt --test-transform-dialect-interpreter
func.func @matmul(%A: tensor<12x9xf32>, %B: tensor<9x6xf32>, %C: tensor<12x6xf32>) -> tensor<12x6xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<12x9xf32>, tensor<9x6xf32>) outs(%C: tensor<12x6xf32>) -> tensor<12x6xf32>
  return %D : tensor<12x6xf32>
}
// use identity layout at function boundaries.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
  transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg1 {bufferize_function_boundaries = true }
}
// result is 连续的memref
func.func @matmul(%arg0: memref<12x9xf32>, %arg1: memref<9x6xf32>, %arg2: memref<12x6xf32>) -> memref<12x6xf32> {
  linalg.matmul ins(%arg0, %arg1 : memref<12x9xf32>, memref<9x6xf32>) outs(%arg2 : memref<12x6xf32>)
  return %arg2 : memref<12x6xf32>
}
// use default at function boundaries.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
  transform.bufferization.one_shot_bufferize %arg1 {bufferize_function_boundaries = true }
}
// result is 非连续的memref(所有func.func的args和返回值均是非连续的）
func.func @matmul(%arg0: memref<12x9xf32, strided<[?, ?], offset: ?>>, %arg1: memref<9x6xf32, strided<[?, ?], offset: ?>>, %arg2: memref<12x6xf32, strided<[?, ?], offset: ?>>) -> memref<12x6xf32, strided<[?, ?], offset: ?>> {
  linalg.matmul ins(%arg0, %arg1 : memref<12x9xf32, strided<[?, ?], offset: ?>>, memref<9x6xf32, strided<[?, ?], offset: ?>>) outs(%arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
  return %arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>
}
```

---

# Conversion

形式：将写好的pattens加入RewriterPatternSet并设置benefit，再apply

```cpp
void runOnOperation() override {
	RewritePatternSet patterns(&getContext());
	patterns.add<xxxx>(patterns.getContext(), /*benefit*/2)
	if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))));
		return signalPassFailure();
}
```

常见的apply形式:

- `applyPartialConversion` ：如果结果是合法（以`ConversionTarget`参数来判断）则保留，如果非法则报错
- `applyFullConversion` ：调用pattern对目标进行转换，直至IR满足`ConversionTarget`设置的目标合法，pattern必须成功才会产生合法的target
- `applyPatternAndFoldGreedily`：尽可能地多次修改，pattern可以失败

前两种常用于dialect conversion，需要多传入一个`ConversionTarget`参数，greedilyConversion一般用于优化pass

## ConversionTarget

常用定义op

```cpp
MLIRContext &ctx = getContext();
ConversionTarget target(ctx);
target.addIllegalDialect<SparseTensorDialect>();
target.addLegalDialect
target.addDynamicallyLegalDialect
target.addLegalOp
target.addDynamicallyLegalOp
```

例如只对标量op进行转换的pattern
```cpp
target.markUnknownOpDynamicallyLegal([](Operation *op) {
	if (isa<math::MathDialect>(op->getDialect()) &&
			llvm::isa<math::LogOp, math::ExpOp,...>(op)) {
	   return op->getResultTypes().front().isa<ShapedType>();
  }
  return true;
});

RewritePatternSet patterns(&ctx);
patterns.add<xxx>(patterns.getContext());
if(failed(applyParticalCpnversion(getOperation(), target,
																	std::move(patterns))))
	return signalPassFailure();
```

ConversionPattern相比RewriterPattern一般多一个[adaptor](#adaptor)参数，用于访问op的opernads

```cpp
// 常用于op的dialect2dialect下降
struct AbsOpToMathAbsConverter : public OpConversionPattern<mhlo::AbsOp> {
  using OpConversionPattern<mhlo::AbsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mhlo::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

// 常用于op的优化行为，也可以用于dialect2dialect中的op下降
struct TransposeSliceLayoutPattern : public OpRewritePattern<mhlo::SliceOp> {
  using OpRewritePattern<mhlo::SliceOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(mhlo::SliceOp op,
                  PatternRewriter &rewriter) const override {
```

---

## dialect conversion

```cpp
mlir/lib/Transforms/Utils/DialectConversion.cpp
```

即dialect_a中的op对应转换到dialect_b中，例如vector dialect → gpu dialect

dialect conversion一般包含op conversion和type conversion

## op conversion

```cpp
mlir/include/mlir/IR/PatternMatch.h
```

1.OpRewritePattern

以vector2gpu为例

```cpp
// mlir/lib/Conversion/ArithToSPIRV/ArithToSPIRV.cpp
// namespace内定义许多op conversion patterns
namespace{
struct ConstantCompositeOpPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OperationConversionPattern;
  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                opAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
	...
	}
}
...
void mlir::populateArithToSPIRVPatterns(RewritePatternSet &patterns) {
	patterns.add<ConstantCompositeOpPattern>(patterns.getContext());
	// 可以设置pattern的/*benefit=*/
	// patterns.add<ConstantCompositeOpPattern>(patterns.getContext(), /*benefit=*/2);
	...
}
} // namespace
```

2.OpInterfaceRewritePattern

专门匹配某种 `OpInterface` 的pattern。例如

```cpp
struct ViewLikeOpXXXPattern
    : public OpInterfaceRewritePattern<ViewLikeOpInterface> {
  ViewLikeOpXXXPattern(MLIRContext *ctx)
      : OpInterfaceRewritePattern<ViewLikeOpInterface>(ctx) {}
  LogicalResult mathAndRewrite(ViewLikeOpInterface viewOp,
                               PatternRewriter &rewriter) const override {
    ...
  }
}
```

## type conversion

```cpp
mlir/Conversion/LLVMCommon/TypeConverter.h
```

对type对改写一般通过 `typeConverter` ，常配合 `ConversionTarget` 使用。其一般包含三个主要函数

- `addConversion` ：定义type转换规则

例如
```cpp
typeConverter converter;
converter.addConversion([&]ToyIntegerType t) -> std::optional<Integer> {
	return Integer::get(&getContext(), t.getWidth())
}
```

- `addTargetMaterialization` ：sourceType→targetType
- `addSourceMaterialization` ：targetType→sourceType
- `addArgumentMaterialization`

```cpp
static Value materializeToXXXCallback(OpBuilder &builder, Type type, ValueRange values) {
  if (xxx)
    ...
  return nullptr;
}

class MyTypeConvert : public TypeConverter {
public:
  MyTypeConvert() {
    addConversion([](Type type)) -> Type {
      if (isSomeType(type))
        return ...;
      return type;
    });
  }

  addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange values) {
    if (...)
      return builder.create<SomeOp>(type, values);
    return nullptr;
  });

  addSourceMaterialization(materializeToXXXCallback);
  addArgumentMaterialization(materializeToXXXCallback);
}
```

---

# Dataflow

MLIR中的数据流图是由Operation和Value构成的：（use-def chain）

- Value的值要么来自于Operation的result，要么来自于BlockArgument
    - 调用getDefiningOp时，BlockArgument会返回null

- 每个Operation的Operand都是到Value的指针

Operation都包含Results和Operands；Results中包含多个OpResult实例，Operands中包含多个OpOperand实例

- 修改Operand时，实际是修改OpOperand，对应value的use-chain也会被修改

## Operation找Value

- getOperands() / getResults()

```cpp
for (auto operand : op.getOperands()) {
	if (auto *def = op.getDefiningOp()) {
	} else {
		// BlockArgument
	}
}
```

- getOpOperands() 用于需要修改operand时

```cpp
IRMapping mapping;
mapping().map(op1.getResults(), op2.getResults());
for (auto &opOperand : op3.getOpOperands()) {
	// 将 op3 的参数里含有 op1 results 的替换为 op2 的
  // lookupOrDefault 指找不到 mapping 就用原来的
  opOperand.set(mapping.lookupOrDefault(opOperand.get()));
}
```

## value找op

- getDefiningOp：可能返回nul

- getUses ：返回OpOperand迭代器，即使用了这个value的OpOperand集合
  - OpOperand &operand : value.getUses()

- getUsers ：返回Operation迭代器，即直接依赖于该value的operation集合
  - user_iterator相当于对use_iterator使用getOwner()
  - use.getOwner() → Operation*

## dataflow framework

见 Analysis 中 dataflow framework 节

---

# DataType

```cpp
mlir/include/mlir/IR/BuiltinTypes.h
```

从ShapedType使用getElementType()获得

类型：

- FloatType
    - getF32
    - getWidth
- IndexType ：target word-size integer
- IntegerType

用法

- 判断类型
    - isInteger
        - isInteger(unsigned width)
    - isIndex
    - isIntOrIndex
    - isIntOrFloat
- 生成 get
    -  RankedTensorType::get(ArrafRef<int64_t> shapes, elemType)
      例如 RankedTenorType newType = RankedTensorType::get({srcDims[0], 1}), srcType.getElementType)
    - IntegerType::get(op→getContext(), 64);

---

# Debug

```cpp
#include "llvm/include/llvm/Support/Debug.h"
LLVM_DEBUG(llvm::dbgs() << "Original loop:\n"
                        << *region->getParentOp() << "\n");
LLVM_DEBUG(llvm::dbgs() << "Checking op: " << *op << "\n");
```

---

# Dianostic

```bash
mlir/docs/Diagnostics.md
mlir/include/mlir/IR/Diagnostics.h
mlir/lib/IR/Diagnostics.cpp
```

当rewrite-pattern使用op的verify(rewrite出的op是否合法)来判断pattern是否match-and-rewrite成功时，那apply-pattern时的报错就是不必要的，可以通过去除handler的办法消除掉这些不必要的报错

使用
```cpp
auto *context = &getContext();
auto handlerID =
    context->getDiagEngine().registerHandler([](Diagnostic &) { return; });
...
RewritePatternSet patterns(context);
patterns.add<xxx>(patterns.getContext());
(void)applyPatternAndFoldGreedily(getOperation(), std::move(patterns));
...
context->getDiagEngine().eraseHandler(handlerID);
```

---

# Dialect

新增一个dialect可以参考最近mlir中新增的[polynomial dialect](https://github.com/llvm/llvm-project/commit/55b6f17071d25b77fcdc910ca9b15f89305137e0) ，然后就是补充各种dialect2dialect的conversion了


## DialectRegistry

The DialectRegistry maps a dialect namespace to a constructor for the matching dialect ：看起来像为dialect中的op外挂新的属性

```cpp
mlir/include/mlir/IR/DialectRegistry.h
```

例如为linalg的op挂上新的interface

```cpp
void mlir::xxx::utils::registerLinalgAggregatedOpInterfaceModel(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LinalgDialect *dialect) {
    linalg::MapOp::attachInterface<MapOpInterface>(*ctx);
    MatmulOp::attachInterface<
        MatmulOpInterface<MatmulOp, linalg::Conv2DNhwcFhwcOp>>(*ctx);
    BatchMatmulOp::attachInterface<
        MatmulOpInterface<BatchMatmulOp, linalg_ext::BatchConv2DNhwcFhwcOp>>(
        *ctx);
    ReduceOp::attachInterface<ReduceOpInterface>(*ctx);
  });
}

// 定义上例如，其中AggregatedOpInterface需要在LinalgExtInterface.td定义
template <typename SrcOpTy, typename DstOpTy>
struct MatmulOpInterface : public AggregatedOpInterface::ExternalModel<
                               MatmulOpInterface<SrcOpTy, DstOpTy>, SrcOpTy> {
  FailureOr<SmallVector<Operation *>>
  decomposeOperation(Operation *op, Operation *value,
                     PatternRewriter &rewriter) const {
	}
};
```

## Affine

1.op

op定义详见 [affine dialect ops](https://mlir.llvm.org/docs/Dialects/Affine/)

- affine.apply
- affine.max / affine.min
- affine.index
- affine.for
- affine.if

op相关的一些函数

```bash
mlir/lib/Dialect/Affine/IR/AffineOps.cpp
```

2.AffineMap

```bash
mlir/inlcude/mlir/IR/AffineMap.h
mlir/lib/IR/AffineMap.cpp
```

- `getFilteredIdentityMap` 创建条件过滤affinemap
```cpp
/// getFilteredIdentityMap(3, [false, false, true]) -> affine_map<(d0, d1, d2) -> (d2)>
AffineMap getFilteredIdentityMap(MLIRContext *ctx, unsigned numDims,
                                llvm::function_ref<bool(AffineDimExpr)> keepDimFilter);
```

- `getPermutationMap` 创建一个permutation的affinemap

```cpp
/// ArrrayRef<int64_t>
static AffineMap getPermutationMap(ArrayRef<unsigned> permutation,
                                  MLIRContext *context);
```

- `getMultiDimMapWithTargets`  创建一个指定输出行为的affinemap，没有计算，只是排序。输入的 `numDims` >= `targets.size()`

```cpp
/// * getMultiDimMapWithTargets(3, [2, 1])
///       -> affine_map<(d0, d1, d2) -> (d2, d1)>
static AffineMap getMultiDimMapWithTargets(unsigned numDims, ArrayRef<unsigned> targets, MLIRContext *context);
```

- bool isEmpty() : Returns true if this affine map is an empty map, i.e., () -> ().

- bool isSingleConstant() :  Returns true if this affine map is a single result constant function.

- int64_t getSingleConstantResult()

- bool isConstant() : Returns true if this affine map has only constant results.

- SmallVector<int64_t> getConstantResults() : Returns the constant results of this map. This method asserts that the map has all constant results.

- unsigned getNumDims() : AffineMap的numDims属性
- unsigned getNumSymbols()
- unsigned getNumResults()
- unsigned getNumInputs()

-  **ArrayRef<AffineExpr> getResults()** 返回每个result的计算affineExpr
- AffineExpr getResult(unsigned idx)

- getDimPosition : 返回result的pos，要求这个idx对应的result是一个 AffineDimExpr。

`AffineDimExpr`  意味着这个result不是计算出来的，一般是等于某个输入，例如affine_map<(d0, d1) -> (d1, d0)>，这个 AffineMap有两个输出，对其getDimPosition(0) = 1, getDimPosition(1) = 0。这个函数一般用在 `permutation` 的 AffineMap 上。

```cpp
unsigned AffineMap::getDimPosition(unsigned idx) const {
  return cast<AffineDimExpr>(getResult(idx)).getPosition();
}
```

- getResultPosition : 返回输入input是当前AffineMap的第几个输出
```cpp
std::optional<unsigned> AffineMap::getResultPosition(AffineExpr input) const {
  if (!isa<AffineDimExpr>(input))
    return std::nullopt;
  for (unsigned i = 0, numResults = getNumResults(); i < numResults; i++) {
    if (getResult(i) == input)
      return i;
  }
  return std::nullopt;
}
```

- isFunctionOfDim

```cpp
/// Return true if any affine expression involves AffineDimExpr `position`.
  bool isFunctionOfDim(unsigned position) const {
    return llvm::any_of(getResults(), [&](AffineExpr e) {
      return e.isFunctionOfDim(position);
    });
  }
```

3.MutableAffineMap

- 可以set一些属性，比如
 `void setResult(unsigned idx, AffineExpr result) { results[idx] = result; }`

- simplify()

使用 `analysis` 简化affinemap，大体是折叠常量相关的计算


4.AffineExpr

```bash
mlir/include/mlir/IR/AffineExpr.h
mlir/lib/IR/AffineExpr.cpp
```

-  AffineExprKind getKind() ： 返回kind

```cpp
  Add,
  /// RHS of mul is always a constant or a symbolic expression.
  Mul,
  /// RHS of mod is always a constant or a symbolic expression with a positive
  /// value.
  Mod,
  /// RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  /// RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,

  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_AFFINE_BINARY_OP = CeilDiv,

  /// Constant integer.
  Constant,
  /// Dimensional identifier.
  DimId,
  /// Symbolic identifier.
  SymbolId,
```

- AffineBinaryOpExpr 继承自 AffineExpr
  - AffineExpr getLHS()
  - AffineExpr getRHS()

- AffineDimExpr
  - unsigned getPosition()

- AffineConstantExpr
  - int64_t getValue()

例:
affine_map (d1, d2) -> (d1 - d2)
这是一个 AffineBinaryOpExpr，kind是add，表达为(1 * d1, -1 * d2)。lhs和rhs都是 AffineConstantExpr，value分别是(1, -1)


``` cpp
/// Return "true" if `candidate` is a negated expression, i.e., Mul(-1, expr).
/// If so, also return the non-negated expression via `expr`.
static bool isNegatedAffineExpr(AffineExpr candidate, AffineExpr &expr) {
  auto mulExpr = dyn_cast<AffineBinaryOpExpr>(candidate);
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;
  if (auto lhs = dyn_cast<AffineConstantExpr>(mulExpr.getLHS())) {
    if (lhs.getValue() == -1) {
      expr = mulExpr.getRHS();
      return true;
    }
  }
  if (auto rhs = dyn_cast<AffineConstantExpr>(mulExpr.getRHS())) {
    if (rhs.getValue() == -1) {
      expr = mulExpr.getLHS();
      return true;
    }
  }
  return false;
}
```

- getPosition()

## linalg

1.op

- linalg.generic
- linalg.fill
- linalg.map{ arith.op / math.op }

```cpp
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, adaptor.getOperands().front(), emptyTensor,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Type elementType = getElementTypeOrSelf(emptyTensor);
          Value operand = args.front();
          Value innerResult =
              elementType.isa<FloatType>()
                  ? rewriter.create<math::AbsFOp>(loc, elementType, operand)
                        .getResult()
                  : rewriter.create<math::AbsIOp>(loc, elementType, operand)
                        .getResult();
          b.create<linalg::YieldOp>(loc, innerResult);
        });
```

- linalg.matmul
- linalg.batch_matmul

2.function

- LinalgInterface
  - bool hasDynamicShape()
  - SmallVector<AffineMap> getIndexingMapsArray()
    ```cpp
    // 判断linalgOp是ElementwiseOp
    auto isElementwiseLinalg = [](linalg::LinalgOp linalgOp) -> bool {
      if (linalgOp.getNumDpsInints() != 1)
        return false;
      return llvm::all_of(linalgOp.getIndexingMapsArray(), [](AffineMap map) {
        return map.isIdentity();
      }) &&
          hasOnlyScalarElementwiseOp(linalgOp->getRegion(0));
    };
    ```

3.LinalgInterface

```bash
mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp
mlir/include/mlir/Dialect/Linalg/IR/LinalgInterfaces.td
```

- getNumLoops() -> unsigned

即返回 getIteratorTypesArray().size()

- getNumParallelLoops

返回 loops 中 parallel轴的数量，这些轴一般可以并行(用`scf.forall`来tile)，而reduction轴都只能用`scf.for`来tile

- getIndexingMapsArray

返回region内的计算。generic op内部是由一堆的计算组成的，即可以看成一个`AffineMap`。

- payloadUsesValueFromOperand

输入是 `OpOperand`，返回这个 `OpOperand` 是否被使用，由此来获得准确 `Memory-Effect`。(inputOperand有user则有read，initOperand必被write，若有user则有read)

例如 https://github.com/llvm/llvm-project/pull/92079/files 中

```cpp
static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    LinalgOp linalgOp) {
  SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
  for (auto [index, operand] : llvm::enumerate(inputOperands)) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(&linalgOp->getOpOperand(index))) {
      effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
  }
  unsigned inputOperandSize = inputOperands.size();

  for (auto [index, operand] : llvm::enumerate(linalgOp.getDpsInits())) {
    if (!llvm::isa<MemRefType>(operand.getType()))
      continue;
    if (linalgOp.payloadUsesValueFromOperand(
            &linalgOp->getOpOperand(index + inputOperandSize))) {
      effects.emplace_back(MemoryEffects::Read::get(), operand, /*stage=*/0,
                           /*effectOnFullRegion=*/true,
                           SideEffects::DefaultResource::get());
    }
    effects.emplace_back(MemoryEffects::Write::get(), operand, /*stage=*/0,
                         /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

```

4.conversion

强烈推荐项目 [triton-linalg](https://github.com/Cambricon/triton-linalg)，大佬们的力作

## scf

```cpp
mlir/lib/Dialect/SCF/IR/SCF.cpp
```

1.op

- scf.for : 循环body必须串行执行，因为每次迭代返回值会写回blockarg，所以下一次使用 blockarg的值受上次迭代的影响
  ```mlir
  %alloc = memref.alloc() : memref<16xi32>
  %1 = scf.for %arg0 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg1 = %alloc) -> (memref<16xi32>) {
    %allco_new = memref.alloc() : memref<16xi32>
    use %arg1
    ...
    scf.yield %alloc_new : memref<16xi32>
  }
  ```

- scf.forall / scf.parallel ：**循环body的程序是可以的并发执行**，没有前后依赖的
  可以使用多线程的方式来执行，线程的id就是循环的迭代变量
  从scf到launch这种转换是可以通过代码自动完成的，需要的额外信息就是每一个循环的轴到launch的轴的映射关系

    ```mlir
    scf.forall (%thread_id_1, %thread_id_2) in (%num_threads_1, %num_thread_2) {
             // ...
          }
        }
    ```

- scf.if

	```cpp
	Block *IfOp::thenBlock() { return &getThenRegion().back(); }
	YieldOp IfOp::thenYield() { return cast<YieldOp>(&thenBlock()->back()); }

	auto cond = op.getCondition();
	auto thenYieldArgs = op.thenYield().getOperands();
	auto elseYieldArgs = op.elseYield().getOperands();
	```

	有一个 `scf.if` 的canonicalize pattern，叫 `ConvertTrivialIfToSelect`，可以尽量消除 else region

	经常在 `bufferize` 后的 `canonicalize` 起效，因为`bufferize` 后 `scf.yield` 的operand更关系更明确了


	```mlir
	// ./build/bin/mlir-opt test_if.mlir --split-input-file --one-shot-bufferize --canonicalize

	// 不能命中，因为thenRegion的yield value属于thenRegion
	// %1 = arith.cmpi slt, %arg1, %c0_i32 : i32
	// %2 = scf.if %1 -> (memref<2xi32>) {
	//   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2xi32>
	//   linalg.map { math.absi } ins(%0 : memref<2xi32, strided<[?], offset: ?>>) outs(%alloc_0 : memref<2xi32>)
	//   scf.yield %alloc_0 : memref<2xi32>
	// } else {
	//   scf.yield %alloc : memref<2xi32>
	// }
	func.func @test_if (%arg0 : tensor<2xi32>, %arg1 : i32) -> tensor<2xi32> {
	  %cst = arith.constant 0 :i32
	  %0 = tensor.empty() : tensor<2xi32>
	  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2xi32>) -> tensor<2xi32>
	  %2 = arith.cmpi slt, %arg1, %cst : i32
	  %3 = scf.if %2 -> tensor<2xi32> {
	    %4 = tensor.empty() : tensor<2xi32>
	    %5 = linalg.map{math.absi} ins(%arg0 : tensor<2xi32>) outs(%4: tensor<2xi32>)
	    scf.yield %5 : tensor<2xi32>
	  } else {
	    scf.yield %1 : tensor<2xi32>
	  }
	  return %3 : tensor<2xi32>
	}

	// -----
	// 可以命中，但不产生select，因为trueVal == falseVal
	// %1 = arith.cmpi slt, %arg1, %c0_i32 : i32
	// scf.if %1 {
	//    linalg.map { math.absi } ins(%0 : memref<2xi32, strided<[?], offset: ?>>) outs(%alloc : memref<2xi32>)
	func.func @test_if (%arg0 : tensor<2xi32>, %arg1 : i32) -> tensor<2xi32> {
	  %cst = arith.constant 0 :i32
	  %0 = tensor.empty() : tensor<2xi32>
	  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2xi32>) -> tensor<2xi32>
	  %2 = arith.cmpi slt, %arg1, %cst : i32
	  %3 = scf.if %2 -> tensor<2xi32> {
	    %5 = linalg.map{math.absi} ins(%arg0 : tensor<2xi32>) outs(%1: tensor<2xi32>)
	    scf.yield %5 : tensor<2xi32>
	  } else {
	    scf.yield %1 : tensor<2xi32>
	  }
	  return %3 : tensor<2xi32>
	}

	// -----
	// 产生select
	// %1 = arith.cmpi slt, %arg1, %c0_i32 : i32
	// %2 = arith.select %1, %alloc, %alloc_0 : memref<2xi32>
	// scf.if %1 {
	//  linalg.map { math.absi } ins(%0 : memref<2xi32, strided<[?], offset: ?>>) outs(%alloc : memref<2xi32>)
	func.func @test_if (%arg0 : tensor<2xi32>, %arg1 : i32) -> tensor<2xi32> {
	  %cst = arith.constant 0 :i32
	  %0 = tensor.empty() : tensor<2xi32>
	  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2xi32>) -> tensor<2xi32>
	  %cst1 = arith.constant 1 :i32
	  %6 = tensor.empty() : tensor<2xi32>
	  %7 = linalg.fill ins(%cst1 : i32) outs(%6 : tensor<2xi32>) -> tensor<2xi32>
	  %2 = arith.cmpi slt, %arg1, %cst : i32
	  %3 = scf.if %2 -> tensor<2xi32> {
	    %5 = linalg.map{math.absi} ins(%arg0 : tensor<2xi32>) outs(%1: tensor<2xi32>)
	    scf.yield %5 : tensor<2xi32>
	  } else {
	    scf.yield %7 : tensor<2xi32>
	  }
	  return %3 : tensor<2xi32>
	}
	```


# TO BE CONTINUE