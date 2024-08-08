---
title: Triton 编译安装
author: tfruan
date: 2024-04-07 20:00:00 +0800
categories: [Triton]
tags: [Triton, Install]
---

# 配置python环境

建议使用conda配置，选择python3.10会稳定些

`conda create -n triton_env python=3.10`

- 根据cuda版本安装pytorch(gpu版)

例如我用的是cuda11.8，那么

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

详细见官网：https://pytorch.org/get-started/previous-versions/

- 安装常见的包

numpy matplotlib pybind11 lit pytest isort pandas tabulate scipy flake8 autopep8

pybind11安装后需要配置环境变量，否则会找不到头文件

```bash
export PYBIND_INCLUDE_PATH=/xxxx/miniconda/envs/triton_env/lib/python3.10/site-packages/pybind11/include
```

下面的源挺好用的 `vim ~/.condarc`

```bash
show_channel_urls: true
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
auto_activate_base: false
```

# 捷径

```bash
git clone https://github.com/triton-lang/triton.git
```

clone llvm 很难搞，如果不用修改源码，就直接安装吧


```bash
pip install git+https://github.com/LLNL/hatchet

pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly --use-deprecated legacy-resolver
```

运行一下

```bash
Python 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import triton
>>> triton.__version__
'3.0.0'


cd triton/python/tutorials/
python 03-matrix-multiplication.py
```


# 编译llvm

```bash
git clone https://github.com/triton-lang/triton.git
git clone https://github.com/llvm/llvm-project.git
```

如果拉取出现下面报错，在repo内输入 `git config --global http.postBuffer 1024288000`

```bash
remote: Compressing objects: 100% (1151/1151), done.
error: RPC failed; result=18, HTTP code = 200| 592.00 KiB/s
fatal: The remote end hung up unexpectedly
fatal: 过早的文件结束符（EOF）
fatal: index-pack failed
```

- 切换llvm commit

git checkout xxx，其中xxx是triton对应的llvm版本号，可以使用 `cat triton/cmake/llvm-hash.txt` 找到

- build

cmake 版本要求3.20以上，记得安装ninja。如果没有root权限就下编译好的二进制，解压后加PATH即可。

```bash
cd xxxpath/llvm-project
mkdir build && cd build

cmake -G Ninja ../llvm \
     -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
     -DLLVM_BUILD_EXAMPLES=ON \
     -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
     -DMLIR_ENABLE_CUDA_RUNNER=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_ENABLE_RTTI=ON \
     -DLLVM_INSTALL_UTILS=ON \
     -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
     -DCMAKE_INSTALL_PREFIX="xxxpath/tools_build/llvm"

ninja -j32
ninja install

cmake --build . --target check-mlir
```

编译时target只能是"X86;NVPTX;AMDGPU"，如果多了riscv，后续编译出的libtriton.so是有问题的，会报错

```bash
ImportError: /lustre/S/ruantingfeng/triton/triton_repo/python/triton/_C/libtriton.so: undefined symbol: LLVMInitializeRISCVAsmParser
```

- 增加环境变量在.bashrc

```bash
export PATH=xxxpath/tools_build/llvm/bin:$PATH
export LLVM_BUILD_DIR=xxxpath/tools_build/llvm
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
```

# 编译triton

```bash
cd xxxpath/triton
conda actiave triron_env
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip install -e python
```

编译好的内容在 `xxxpath/triton/python/build` 中

而且 `libtriton.so` 已经加到 `_c` 中了

```bash
$ ll python/triton/_C/
include/      libtriton.so
```

再加个环境变量

```bash
export TRITON_HOME=/lustre/S/ruantingfeng/triton/triton_repo
export PYTHONPATH=$TRITON_HOME/python:${PYTHONPATH}
```

测试一下，没啥问题就可以运行 `python/tutorials` 中的测试（跑了一下03-matrix-multiplication.py，看起来暂时干不过cuBLAS）

```bash
$ python
Python 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import triton
>>> triton.__version__
'3.0.0'
>>>
```