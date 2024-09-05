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
