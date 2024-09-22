以 GPU 为 Target 的 MLIR Codegen Pipeline components

# Pipeline

![dialect](/assets/img/blog/img_mlir_gpu_pipeline_component/nvvm_dialect_ir.png)

## some ways

## Trion-Lang

# components

## mlir office Dialect

### NVVM Dialect

- docs: <https://mlir.llvm.org/docs/Dialects/NVVMDialect/>
- file: <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/LLVMIR/NVVMOps.td>

NVVM Dialect 本质上是也是 LLVM Dialect IR，毕竟都定义在 `mlir/Dialect/LLVMIR/` 目录下了。LLVM Dialect 提供了与 LLVM IR 对应的操作和类型，使得 MLIR 可以表达 LLVM IR 的语义，方便 MLIR 与 LLVM 生态地桥接。

NVVM Dialect 和 [NVVM IR](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsNVVM.td) 是两个概念。NVVM IR 是一种 LLVM IR，额外封装了一些 hardware special 的 `convention` 和 `intrinsic` 等。NVVM Dialect 和 LLVM Dialect 之间的关系与 LLVM IR 和 NVVM IR 之间的关系差不多。

这一层的原语表达接近 PTX 的语义，期望是能够直接对应 PTX 指令。但 NVVM Dialect 当前对 PTX model 的支持还不完备，并且 NVVM Ops don't include PTX or SM versions unlike LLVM intrinsics。还会继续添加相关op。

当前 MLIR 设置中是依托 [BasicPtxBuilderInterface](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/BasicPtxBuilderInterface.cpp) 来直接从 tablegen op 中直接生成 PTX 指令。

> 而 Triton 使用自己的 PTXNBuilder 来生成 inline PTX asm，因为当前 Triton 对相关的场景需求较小，所以暂时没有在 pipeline 中引入 NVVM Dialect 的想法。

### NVGPU Dialect

- docs: <https://mlir.llvm.org/docs/Dialects/NVGPU/>
- file: <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/NVGPU>

NVGPU Dialect 是 NVVM Dialect 的更高级抽象。主要是 device ops，最终 nvgpu-nvvm-ptx。

例子来自：<https://github.com/llvm/llvm-project/issues/69448>

```text
// nvgpu dialect ir
%matrixD = nvgpu.wargroup.mma %wgmmaDescA, %wgmmaDescB, %matrixC {transposeB}:
%matrixD = nvgpu.wargroup.mma %wgmmaDescA, %wgmmaDescB, %matrixC {transposeB}:
!nvgpu.wgmma.descriptor<tensor = memref<128x64xf16, 3>>,
!nvgpu.wgmma.descriptor<tensor = memref<64x128xf16, 3>>,
!nvgpu.wargroup.accumulator<fragmented = vector<128x128xf32>>
->
!nvgpu.wargroup.accumulator<fragmented = vector<128x128xf32>>

// ptx
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f1, %f2,...}, %dA, %dB, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f1, %f2,...}, %dA+2, %dB+128, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f1, %f2,...}, %dA+4, %dB+256, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f1, %f2,...}, %dA+8, %dB+348, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f500,%f501,...} %dA+512, %dB, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f500,%f501,...} %dA+514, %dB+128, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f500,%f501,...} %dA+516, %dB+256, p, 1, 1, 0, 1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f500,%f501,...} %dA+518, %dB+348, p, 1, 1, 0, 1;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
```

### GPU Dialect

- docs: <https://mlir.llvm.org/docs/Dialects/GPU/>
- file: <https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/GPU>

提供 launch kernel、kernel invocation 相关的抽象。主要是 host ops，主要下降到 runtime calls。

## Triton

- Triton TritonNvidiaGPU -> Upstream NVGPU
- Triton NVGPU -> Upstream NVVM (almost 1to1 to PTX)

### TritonGPU Dialect

- file: <https://github.com/triton-lang/triton/tree/main/include/triton/Dialect/TritonGPU>

### TritonNvidiaGPU Dialect

- file: <https://github.com/triton-lang/triton/tree/main/include/triton/Dialect/TritonNvidiaGPU>

### Triton NVGPU Dialect

- file: <https://github.com/triton-lang/triton/tree/main/third_party/nvidia/include/Dialect/NVGPU>

和当前upstream [NVGPU Dialect](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/NVGPU/IR/NVGPU.td)非常不同。作用上类似 upstream 的 NVVM Dialect，抽象了一些操作随后会将其转为 inline asm。

理论上 LLVM 中如果支持了等价的 intrinsics，这里就会被移除。

### TritonAMDGPU Dialect

- file: <https://github.com/triton-lang/triton/tree/main/third_party/amd/include/Dialect/TritonAMDGPU>

# 参考资料

- [mlir docs](https://mlir.llvm.org/docs/)
- [triton-lang](https://github.com/triton-lang/triton)
- [2024 EuroLLVM - Zero to Hero: Programming Nvidia Hopper Tensor Core with MLIR's NVGPU Dialect](https://www.youtube.com/watch?v=V3Q9IjsgXvA)
