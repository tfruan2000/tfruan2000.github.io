---
title: CPP Note Needed for MLIR
author: tfruan
date: 2024-08-06 12:00:00 +0800
categories: [MLIR]
tags: [MLIR, Coding]
---

# 内存模型

C++程序在执行时，将内存大方向划分为**4个区域**

- 代码区：存放函数体的二进制代码，由操作系统进行管理的
  - 代码区是**共享**的，共享的目的是对于频繁被执行的程序，只需要内存中有一份代码即可
  - 代码区是**只读**的，使其只读的原因是防止程序意外地修改了它的指令

- 全局区：存放**全局变量和静态变量**static以及常量，程序作用后释放

- 栈区：由编译器自动分配释放, 存放函数的参数值,局部变量等 **(函数作用后释放)**

程序执行就是进入和退出一个个作用域的过程，这个过程使用栈实现，当进入一个新作用域时会压入一个栈帧，**栈帧包含当前作用域的所有局部变量**，当退出这个作用域时，栈帧弹出。

如果这个作用域内有定义指针，这个指针也是临时变量，会指向堆上的一片内存，需要在作用域结束前手动释放。

- 堆区：由程序员分配new和释放,若程序员不释放,程序结束时由操作系统回收 **(程序周期结束后释放)**

# 关键字

- const

const 表示常量。在成员函数声明和定义中，const 关键字表示该函数是一个常量成员函数，即**该函数不会修改对象的成员变量**。

const 函数中可以修改用 mutable 修饰的成员变量，以此来修改一些和类状态无关的成员变量。但不要直接把const取消，可能会导致对象在被认为是不可变的情况下被意外修改。

`const char *` -> const 修饰的是 char，意味着指针指向的地址中的值不能被修改，但是可以更改指针指向的地址

`char * const` -> const 修饰的是 char *，意味着指针指向的地址中的值可以被修改，但是不可以更改指针指向的地址

- constexpr

constexpr: 强调在编译时一定已知，常用在编译过程中指导优化。例如字面量 `StringLiteral` 就需要配合 `constexpr` 使用。

const: 只强调不能被改变，可能在运行时才可知。const 变量可以直接找到其地址，修改地址里面的值，但建议

- static

static 表示静态变量，使用该关键词控制变量/函数的可见域，只在该文件内部起作用

- final

final 用于**防止类被继承**。当类被声明为 final，表示该类不能被其他类继承。

- public

public 是 C++ 中的访问修饰符之一，用于指定类的成员的访问权限。public 成员可以被任何类或函数访问。表示继承的成员和方法在派生类中是公共的，可以被外部访问。

- virtual

virtual修饰的函数称为虚函数，子类可以直接用或者override。纯虚函数函数最后有=0，子类必须override。

解决菱形继承问题：virtual public

- override

override 是 C++11 引入的关键字，用于**显式地声明一个成员函数覆盖了基类的虚函数**。

在这里，这样做有助于提高代码的可读性和可维护性，同时也可以帮助编译器检查函数是否正确地覆盖了基类的虚函数。

- define

宏定义define相当于内存替换，不分配内存，可以通过undefine取消。

- inline

可以把简单函数直接写到头文件中。类中的函数除了virtual，都会自动inline

virtual也可以inline，只要没有多态（重载也算一种多态）

inline函数比define函数更高效，因为不用参数压栈、栈帧管理和结果返回。提高了运行效率，而且会做安全检查。但是多次调用情况下会导致代码膨胀。

define 和 inline 的区别

- 内联函数在编译时展开，宏在预编译时展开
- 内联函数直接嵌入到目标代码中，宏是简单的做文本替换
- 内联函数可以完成诸如类型检测，语句是否正确等编译功能，宏就不具有这样的功能
- 宏不是函数，inline函数是函数
- 宏在定义时要小心处理宏参数，一般用括号括起来，否则容易出现二义性。而内联函数不会出现二义性

- volatile

volatile 声明的类型变量表示可以被某些编译器未知的因素更改，编译器对访问该变量的代码就不再进行优化，从而可以提供对特殊地址的稳定访问

- explicit

防止隐式转换和复制初始化

# 深浅拷贝

构造函数分为默认、带参、拷贝构造（函数参数、非引用返回、初始化对象）。浅拷贝：没有显示定义拷贝

为了避免浅拷贝时，需要自己写拷贝构造函数。需要把值复制给新的对象，而不是把权限给新对象

函数参数传递对象（值传递）就是浅拷贝

类中有指针（包括虚函数）的情况下，一定需要拷贝构造函数

浅拷贝可能造成double-free

# 智能指针

自动管理内存，在离开作用域时释放内存，避免内存泄露。

- std::unique_ptr : **独占所有权的智能指针**，确保同一时刻只有一个指针可以拥有和访问资源，当其被销毁或者重制时，会自动释放资源。适用于管理单个对象（pass）或动态分配的数组。

`make_unique`是一个函数模板，用于创建并返回一个`std::unique_ptr`智能指针

```cpp
std::unique_ptr<OperationDefinition> def =
 std::make_unique<OperationDefinition>(op, nameLoc, endLoc);
```

- std::shared_ptr: 共享所有权的智能指针，允许多个指针共享同一个资源，使用引用计数来跟踪资源的所有权，当最后一个shared_ptr被销毁或重制时，资源才会释放。适用于跨多个对象共享资源

例：所有 `canonicalize` 行为在进行时都共享 `patterns`

```cpp
// mlir/lib/Transforms/Canonicalizer.cpp
GreedyRewriteConfig config;
std::shared_ptr<const FrozenRewritePatternSet> patterns;
```

- std::weak_ptr: 作为std::shared_ptr的辅助类，允许观察和访问由std::shared_ptr管理的资源，但不会增加引用计数。用于解决std::share_ptr造成的循环引用，使用其允许你创建一个指向由`std::shared_ptr`管理的资源的非拥有（弱）引用，而不会增加引用计数。它通过解除`std::shared_ptr`的循环引用来避免内存泄漏。

# lambda编程

```cpp
// [] : 捕获列表，可以是值捕获、引用捕获或者不捕获任何变量
[capture clause](parameters) -> return_type {
    // Lambda函数体
    // 可以访问外部变量，参数等
    return expression; // 可选
};
```

用 `[&]` 可以捕获外面的值，如果lambda函数内使用外面的值较少，可以直接加在 `[]` 内

最好指定输出格式

```cpp
auto getReassociations = [&](const DenseSet<int64_t>& dimIndexSet) -> SmallVector<ReassociationIndices> {
auto getNewPermutation = [](const SmallVector<int64_t>& relativeOrder) -> SmallVector<int64_t> {
```

```cpp
llvm::for_each(relativeOrder, [](int64_t i) {llvm::outs() << i << " ";});

llvm::all_of(llvm::zip(array, array.slice(1)), [](const auto& pair) {
   return std::get<0>(pair) <= std::get<1>(pair);
});

llvm::find_if(shapeIndexs, [&](int64_t shapeIndex) {
   return !oneSizeDimIndexsSet.count(shapeIndex);
});
```

# assert

assert(a && “debug info”)

a一般为bool表达式，当a的结果为false时，输出”debug info”

# 类中重载

- 重载操作符

```cpp
class AliasResult {
public:
  enum Kind {
    NoAlias = 0,
    PartialAlias,
    MayAlias,
    MustAlias,
  };
  AliasResult(Kind K) : kind(K) {};
  bool operator==(const AliasResult &other) const { return kind == other.kind; }
  bool operator!=(const AliasResult &other) const { return !(*this == other); }
private:
  Kind kind;
};
```

- 重载函数

```cpp
class baseA {
public:
  virtual bool task(xxx) {
    ...
    return ..
  };
}

class A : public baseA {
public:
  bool task(xxx) override {
    ...
  }
}
```

# 左值 右值

左值用于写操作，可以存储数据。典型的左值有:变量、数组元素、指针等。

右值用于读操作，读到的数据放在一个看不见的临时变量。典型的右值有:字面量(literal)、运算产生的临时对象等

**区别**：

- 地址:左值有地址,右值没有地址
- 生命周期:左值在程序的多个位置可使用,右值表达式计算完成就消失
- 赋值操作:左值可作为赋值操作的左操作数,右值只能作为赋值操作的右操作数。即左值可以被修改，而右值不能。

# move and forward

std::move和std::forward都是执行强制转换的函数。std::move 是无条件将实参转换成右值， std::forward 则仅在某个特定条件满足时执行同一个强制转换

- move

类作为函数参数时，若直接传递(一般是const &)，则是调用拷贝构造

拷贝构造的一般流程：先复制临时变量，再把复制内容放到目的内存，回收临时

优化：将 目的地址指针 和 临时变量指针  指向直接交换，然后销毁临时变量指向的内存。这就完成了转移所有权，即move

转移所有权：将一个容器的所有权转移给另一个容器

```cpp
std::vector<int> source = {1, 2, 3, 4, 5};
std::vector<int> destination = std::move(source); // 使用 std::move 转移所有权
// 现在 source 是一个空的 vector
std::cout << "Source size: " << source.size() << std::endl; // 输出 0
// destination 包含了原始 vector 的内容
std::cout << "Destination size: " << destination.size() << std::endl; // 输出 5
```

实现代码：使用 `remove_reference` 擦除 T 的引用类型，从而保证该函数返回的一定是右值引用

```c++
template <class _Tp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __libcpp_remove_reference_t<_Tp>&&
move(_LIBCPP_LIFETIMEBOUND _Tp&& __t) _NOEXCEPT {
  typedef _LIBCPP_NODEBUG __libcpp_remove_reference_t<_Tp> _Up;
  return static_cast<_Up&&>(__t);
}
```

- forward

move强制把左值转变为右值引用，forward就是保持参数传递的左右值特性

当传入rvalue（临时变量），则等效为`std::move`。如果外面传来了lvalue, 它就转发lvalue并且启用复制. 然后它也还能保留const.

```cpp
void A::set(const T &val)
  m_var = var;  //copy

void A::set(T &&val)
  m_var=move(val)
```

理论上forward cover了move的所有情况，但需要额外带一个模版参数T(处理临时变量用右值引用`string &&`, 处理普通变量用const引用`const string &`)，可读性差点。而且这两者都可以被static_cast替代

实现：当传入值为右值引用时才执行向右值类型的强制类型转换

```cpp
template <class _Tp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _Tp&&
forward(_LIBCPP_LIFETIMEBOUND __libcpp_remove_reference_t<_Tp>&& __t) _NOEXCEPT {
  static_assert(!is_lvalue_reference<_Tp>::value, "cannot forward an rvalue as an lvalue");
  return static_cast<_Tp&&>(__t);
}
```

# emplace_back 和 push_back

`push_back` 和 `emplace_back`的一个区别在于实例化 `std::vector<T>` 之后，push_back的参数是已知的，就是 `T` ; 而 `emplace_back` 的参数是未知的，需要从参数中推导出来。

`push_back` 会在合适的位置使用传入的常量左值引用或者右值引用直接构造新元素。`emplace_back` 除此之外可以直接传入构造函数。

`emplace_back` 支持除了支持 `copy` 和 `move` 构造之外，还支持直接构造。如果**类型不支持move构造，并且copy构造代价比较大**的时候，使用emplace_back是可以得到更多的性能收益的。

`emplace_back` 过程中使用完美转发，所以不断在转发大小不相同的char数组。`push_back`可以使用同一段代码，`emplace_back`每一次调用都需要更新新的代码。

# 移位操作

算术移位和逻辑移位

- 算术右移(`<<`)会保留符号位，右移时符号位会继续复制到最新的最高位 (arith.shrsi)

- 逻辑右移不保留符号位，高位补0 (arith.shrui)

- 算术和逻辑左移相同 (arith.shli)

# 构造函数

带参构造函数（右值、常量左值引用）

赋值构造

构造函数不能是虚函数

# include guard

`pragma once` 防止头文件被引用多次，保证头文件只被编译一次

mlir的项目中，头文件的写法都有传统的include guard，用 `ifdef` 的方法代码移植性更好。

```cpp
#ifdef TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
#define TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H

...

#endif TRITON_LINALG_CONVERSION_ARITHTOLINALG_ARITHTOLINALG_H
```
