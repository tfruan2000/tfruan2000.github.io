# CPP

1.虚函数是加virtual修饰的函数，子类可以直接用或者override。纯虚函数函数最后有=0，子类必须override

2.const inline static define volatile

宏定义define相当于内存替换，不分配内存，可以通过undefine取消

const常量修饰符 。const在数据段，define在代码段，static在静态区

static表示该变量或函数只能在该文件内起作用，运行前就分配好。比如一个只写在cpp中的函数就不用在对应的头文件中声明了。

inline可以把简单函数直接写到头文件中。类中的函数除了virtual，都会自动inline

virtual也可以inline，只要没有多态（重载也算一种多态）

inline函数比define函数更高效，因为不用参数压栈、栈帧管理和结果返回。提高了运行效率，而且会做安全检查。但是多次调用情况下会导致代码膨胀

volatile让编译器别优化该变量。本来优化后会从寄存器中读，但阻碍优化就会每次都从内存读

3.构造函数分为默认、带参、拷贝构造（函数参数、非引用返回、初始化对象）。浅拷贝：没有显示定义拷贝

为了避免浅拷贝时，需要自己写拷贝构造函数。需要把值复制给新的对象，而不是把权限给新对象

函数参数传递对象（值传递）就是浅拷贝

类中有指针（包括虚函数）的情况下，一定需要拷贝构造函数

浅拷贝可能造成double-free

4.move

类作为函数参数时，若直接传递(一般是const &)，则是调用拷贝构造

拷贝构造的一般流程：先复制临时变量，再把复制内容放到目的内存，回收临时

优化：将 目的地址指针 和 临时变量指针  指向直接交换，然后销毁临时变量指向的内存。。这就完成了转移所有权，即move

`T &` -> `T &&`

5.forward

move强制把左值转变为右值引用，forward就是保持参数传递的左右值特性

forward处理左值常量引用时，直接copy拷贝构造；处理右值时，直接走move

```cpp
void A::set(const T &val)
  m_var = var;  //copy

void A::set(T &&val)
  m_var=move(val)
```

理论上forward cover了move的所有情况，但需要额外带一个模版参数T(处理临时变量用右值引用`string &&`, 处理普通变量用const引用`const string &`)，可读性差点。而且这两者都可以被static_cast替代

6.多用++i

i++包含 取值、加法、赋值三个操作，在高并发时可能出问题

7.struct和class

struct默认是public，class默认是private

8.explicit

防止隐式转换和复制初始化

9.构造函数

带参构造函数（右值、常量左值引用）

赋值构造

构造函数不能是虚函数

11.菱形继承问题：virtual public

12.const constexpr

constexpr: 在编译时一定已知

const: 只强调不能被改变，可能在运行时才可知

但是 const 变量可以直接找到其地址，修改地址里面的值。

const 成员函数中可以用 mutable 改变成员变量，但不要直接把const取消，因为const函数可以修改一些和类状态无关的成员变量。

13.算术移位和逻辑移位

- 算术右移(`<<`)会保留符号位，右移时符号位会继续复制到最新的最高位 (arith.shrsi)

- 逻辑右移不保留符号位，高位补0 (arith.shrui)

- 算术和逻辑左移相同 (arith.shli)

14.作用域

比如 if、func这些，块内声明的变量作用域局限在块内。一些程序语言允许块内和快外同名，但有些不可以(java)

15.栈帧

程序执行就是进入和退出一个个作用域的过程，这个过程使用栈实现，当进入一个新作用域时会压入一个栈帧，栈帧包含当前作用域的所有局部变量，当退出这个作用域时，栈帧弹出。

如果这个作用域内有定义指针，这个指针也是临时变量，会指向堆上的一片内存，需要在作用域结束前手动释放。

16.python中的super

基础用法：重载了父类的函数后，使用super调用父类的该函数。实际上python是使用MRO列表中的下一个对应函数来返回。

17.include guard

`pragma once` 防止头文件被引用多次，保证头文件只被编译一次

mlir的项目中，头文件的写法都有传统的include guard，用 `ifdef` 的方法代码移植性更好。

```cpp
#ifdef TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
#define TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H

...

#endif TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
```

18.智能指针

```cpp
// 头文件中声明
std::unique_ptr<Pass>

// cpp中定义pass
std::make_unique<Pass>
```

# arch

1.barrier是核间的，sync是核内的

2.workgroup其实是onencl的术语，对应cuda就是逻辑上的block

3.一个kernel有一个grid，一个grid有多个block，一个block有多个thread（按warp）分组

每个Grid可以最多创建65535个block，每个Block最多512个thread

访问层次上，thread操作register(local memory)，block操作share memory，grid操作global ram

> 所以 thread 一次只能取一个寄存器的值。
>
> 每个 bank 一次只能处理一个访问，当多条 thread 访问的数据处于同一个 bank，则会发生 bank conflict。
> 访存合并是将多个 transactions 合并成较少的 transactions ，但每个线程还是只访问一个元素数据。
>
> 因为一个 warp 中的 thread 在同一时间执行相同的指令。所以访存连续后，可以加大单时间内访问的数量

一个block内的所有thread通过share memory（这个share mem 就是负责该block的L1 Cache）来交换数据。不同block之间的thread是无法通信的

（但新架构上，block之间可以共用一片shared mem）

越高访存层次的访问latency越小

4.把一个个block分配给SM进行运算；而block中的thread又会以warp（线程束）为单位，对thread进行分组计算。目前CUDA的warp大小都是32，也就是说32个thread会被组成一个warp来一起执行。同一个warp中的thread执行的指令是相同的，只是处理的数据不同

基本上warp 分组的动作是由SM 自动进行的，会以连续的方式来做分组。比如说如果有一个block 里有128 个thread 的话，就会被分成四组warp，第0-31 个thread 会是warp 1、32-63 是warp 2、64-95是warp 3、96-127 是warp 4。而如果block 里面的thread 数量不是32 的倍数，那他会把剩下的thread独立成一个warp

对于计算密集型kernel，4-warp配置通常是最常用/首选的。8-warp配置可能会在prologue或epilogue阶段引入一些延迟。对于非常小的GEMM问题，可以尝试2-warp或1-warp配置。

一个SM 可以处理多个线程块block，当其中有block 的所有thread 都处理完后，他就会再去找其他还没处理的block 来处理。

5.thread block cluster

一个Block的Thread配较少时->执行慢

一个Block的Thread配较多时->SM利用率低，且block可使用的smem有限，当负责的任务规模较大时，则需要使用gdram，使得IO成为瓶颈

以block为粒度来处理任务阻碍运行效率，需要提供更大粒度的线程组

thread block cluster是更大粒度的线程组，其中的线程组可以访问分布在不同block的smem，这些smem称为 distributed smem

smem和SM的L1Cache共用一块物理空间，但Cache是不可见的，而smem是程序员可见的，这是SM私有（L2 Cache是共用的）。但若thread block cluster的block在多个SM上运行，就需要特殊的结构实现SM的smem共享。

Hopper在L1和L2之间加了SM-2-SM Network，实现SM1可以访问SM2的L1 Cache

6.gdram-smem memcopy的流程：

(1)虚拟地址转物理地址
(2)ROB申请空间，以便于(乱序执行时)提交时获得正确的目的地址
(3)发送总线请求到LLC
(4)LLC命中则返回，失败则从gdram中找

7. PTX 和 SSAS

ptx 和 sass 的区别在于，一个表示虚拟，一个表示实际。 ptx 指令只定义了指令的功能，而 sass 指令表明了针对不同代架构的底层实现。
> cubin 是硬件二进制指令。

8.CTA, CGA

cooperative thread array：就是 thread block

cooperative grid array：相当于一组 CTA

9.DMA, TMA

- Direct Memory Access 用于提高数据传输
- Tensor Memory Accelerator 专门设计用来张量数据的传输

10.memory-coalecse

如果一个 warp 需要处理的数据连续时，就可以将多个 thread 的内存访问请求合并成一个或者较少个内存传输请求，而不是每个 thread 单独进行内存访问。

pass 前后都是每条 thread 单独发送自己的内存请求，这些请求被合并后由于 LSU(Load/Store Uint) 发送请求到硬件。

11.layout swizzling

smem 中数据是以 bank 的形式组织的，每个 bank 可以单独处理一个内存访问请求。如果多个内存访问请求都指向同一 bank，则会产生 bank-conflict。

所以需要 layout-swizzling 对数据进行重新排列，确保warp在同一时刻的多个内存访问请求处于不同的 bank。

# MLIR

1. SmallVectorImpl

写一个以SmallVector为参数的函数，如果传入的元素个数是固定的，建议使用`SmallVectorImpl` 作为形参，来避免**对堆栈元素的隐式数量进行硬编码**。因为 SmallVector有一个参数N表示元素个数，直接使用SmallVectorImpl能避免该行参的拷贝

2. triton通过Layout来表征Thread对数据的访问模式，信息通过layout这种attr往下传递，指导下降

3. MLIR提供了两个tablegen模块，即：ODS和DRR。
ODS：统一Dialect，Operation等Dialect内部类的创建；DRR：统一Canonicalization, Transformation和Conversion的创建，即PatternRewritter的管理（除此之外，也提供对Pass的管理）。

4. canonicalize

The goal of canonicalize is to make analysis and optimization more efficient, performance improvement is not the goal

Canonicalize shouldn't lose the semantic of original operation:

5. 优化

- broadcast + transpose -> transpose -> broadcast

- transpose + transpose -> transpose
  - 前后 transpose 之间再夹着某些 op(map)

- fold unit dim (把需要在片上进行计算的op退化为shape变化)
  - broadcast(3200 -> 1x3200) -> collapse_shape + expand_shape
  - reduce(1x3200x1 -> 3200) -> collapse_shape + map + expand_shape

6. forall

torch 中的 launch func 显示地作为了 host func，而 grid 则映射到任务应该如何切分。

```text
func {
  ops1 // gmem
  forall1 {
    ops2 // smem
    forall2 {
      ops3
    }
  }
}
```

7.mlir codegen

mix mlir dialect -> llvm dialect -> llvm ir -> hardware intrinsics -> hardware assembly

> intrinsic 其实是 LLVM 中一些根据硬件后端预定义的 vairable 和 function。

mix mlir dialect -> llvm dialect -> -> llvm ir -> ptx assembly -> 通过cuda-rt binary 转为 sass ，而不用转为 intrinsics

8.应用场景

mlir的codegen更适合访存密集性任务，因为比较好优化不同memory-space之间的data flow。

对 SIMD 硬件的优化 和 SIMT 硬件的优化

- SIMD
  - latency bound 优化，越快完成越好 -> 保证访存连续性，用连续指令(非strided，非scalar)
  - 其他优化：
    - tile(+fuse) 到不同 core 上并行执行，core之间利用smem交换数据 -> 减少 data move
    - 在core内循环展开(最内维)做软流水；core之间async -> 减少访存 latency
- SIMT
  - throughtput bound 优化，吞吐越大越大 -> 用好 DMA 和 TMA，打满 tensorcore
  - 其他优化：
    - 离散访存优化 smem 中 memory-coalesce（warp内的thread在同一时刻执行的指令是相同的，所以要减少指令的下发）、layout-swlizzed -> 减少访存 latency
    - 异步调度 warp，通过wrap切换来掩盖访存延迟(其实相当于软流水) -> 减少访存 latency

> core 之间 async  <--> warp 之间 async
>
> 访存优化三板斧：减少数据搬运(用上smem，都写完然后IO出去)，减少数据访存延迟(软流水+减少bank conlict)，保证负载均衡

# LLM note

## attention

1.QK的相似度计算不用除以模长

点积直接表现了向量之间的相似性，在后序softmax计算中，越大的点积就能越突出（越大的权重）。

如果除以模长，就变成了余弦相似度，忽略了向量长度因素的影响。

在self-attention和MHA中，Q和K都是通过变换而来的，保留模长的信息保证了在训练中可以改变变换Q和K的变换参数。

所以处理模长当前只增加了计算复杂度，并未带来显著的收益。

2.除以 $\sqrt{d}$ 得作用

缩放因子用于平衡不同维度的影响，**使得softmax输出更加平滑**，防止输出过于尖锐，导致梯度消失/爆炸。

选择 $\sqrt{d}$ 而不是 $d$ 作为缩放因子，是一种权衡，既保证了点积的值不会太大，也保证了softmax中有一定区分度。

3.softmax激活函数

softmax会将最后的结果归一化，相当于概率分布(落在0～1之间)，全部的结果加起来为1。

这反应的是每个Key对Query的重要性。

其他激活函数不能满足这两点，或者要增加额外的计算，而且softmax梯度计算简单。

4.mask

attention(decoder) 中的 mask：因为是对初始输入一个个计算，当对token_i进行计算时，需要舍弃掉其后的token，所以需要mask来做舍去的行为(softmax时会认为-inf的部分是0)，**以排除干扰**

encoder的设计初衷是给 decoder 提供完整输入序列的 feature 表示。而 decoder 会逐 token 输出，只在乎时间 i 及之前的 token 信息。

5.kvcache

以空间换时间。
window attn：每个token只和包含本身在内的前n个token做attn（使用前几次decode的kvcache），这样kvcache就可以只存一个window的个数。更早的token信息可以跨层流动，就像cnn中的感受野（输出只跟一些输入有关）

> window attn 是在 Mixtral 中提出的，Mixtral 中Rotary的意思就是：通过某种规则，将Cache中的数据旋转回正确的位置，以便能正确做Attention。因为cache每次都在更新，重新写入，所顺序是不符合期望的。

kv cache存储会造成大量碎片化 -> 使用分页管理(page):将每个序列的键值划分为块，采用非连续的存储分配方案，减少空间浪费

6.MQA、GQA、MLA

- MQA：所有 Attention-Head 共享一组 KV Cache
- GQA：一组 KV Cache 支持一个 Group 的 Q
- MLA：不再做 QK 的乘积，直接使用投影表示。无法使用RoPE位置编码，而使用ALIBI这种。

## 推理

1.prefill 和 decode

推理过程分为prefill和decode，只不过decode是逐个生成token，不能像prefill那样大段prompt做并行。

- prefill：模型理解用户的输入，需要吃下完整一段数据
  - 两个目标：生成 Prompt token 时的 KV Cache + 生成首个 token
  - Q K V 序列长度相同，一般是 计算密集型(计算瓶颈)
  - 常使用 Flash-Attention 算子优化这方面
- decode：模型逐token生成回答(续写)
  - 在原始序列上继续(续写)生成 token
  - Q 一般远小于 K 和 V，是 访存密集型(访存瓶颈)，每次生成 token 都要去读已经存储的 KV Cache
  - 常使用 Page-Attention 的技术去优化 KV Cache 的存储。

2.decoder only

当前LLM的架构基本都是decoder only，好处是训练效率高：方便并行（特别是流水并行），加大参数量更简单

3.量化

量化：低位宽数据代替高位宽数据，激活值很难量化（存在异常值，导致量化后精度损失严重，所以量化系统不能全模型统一，按照量化难易程度进行分块）

4.在线推理和离线推理

- 在线推理
  - 需求：实时请求，高并发
  - 技术：动态批处理(合理调整batch，连续批处理)，内存优化(kv cache管理)
  - vLLM
- 离线推理
  - 需求：极致性能，单卡内存
  - 技术：量化、静态批处理
  - TensorRT-LLM

## optim

1.算子融合

- 训练：减少内存带宽占用。需要考虑梯度因素的印象，所以有些中间结果必须要保存。需要考虑**混合精度训练**等带来的影响。
- 推理：减少内存读写，提升计算图执行速度。
