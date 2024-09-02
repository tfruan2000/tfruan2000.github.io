# 从一个 MLIR Programer 的角度来读 OpenAI Triton 源码

[Triton](https://github.com/triton-lang/triton/tree/main) 是什么？这个问题很多大佬都已经回答过了，阅读完以下 blog 相信大家会有个基础的理解。

- 杨军老师的 [谈谈对OpenAI Triton的一些理解](https://zhuanlan.zhihu.com/p/613244988)，帮助大家建立一个宏观印象
- 董鑫大佬的 [如何入门 OpenAI Triton 编程?](https://www.zhihu.com/question/622685131/answer/3217107882)，帮助了解更多关于语法上的信息
- BobHuang大佬的 [浅析 Triton 执行流程](https://zhuanlan.zhihu.com/p/712640431)，帮助初学者大致明白一段 python 程序如何进入 triton pipeline，然后跑起来。

有一些基础的认识后，大家就可以在 Triton 中各取所需了。作为一个 MLIR Programer，我还希望了解每个 transform pass 和 每一步 conversion 做了什么事情，所以本文是个人读源码的一个记录。

> 当前 Triton 中关于 arch 和 non-arch 的抽象界限比较模糊，所以 transform 和 conversion 中可能含有很多 hardware-special 的信息。

本文