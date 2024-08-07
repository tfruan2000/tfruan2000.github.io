---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

我的CV: [tfruan's CV](../files/cv_rtf.pdf)

Education
======
* M.S. 中国科学院计算技术研究所, 2022-2025
* B.S. 西安交通大学, 2018-2022

Project Experience
======
* Triton kernel性能优化 [2024.06-至今]
  * 浅层优化：通过替换算子、合并kernel、拆时间片循环等方式实现初步优化。
  * 深层优化：分析下降所得IR，使用perf工具，对照算子库实现等方式，优化kernel的下降行为。

* 基于MLIR的AI编译器开发 [2023.08-至今]
  * 性能优化：为linalg和triton实现ext-canonicalize；实现layout transpose等pass以优化IR下降。实现continuity-enhancement以避免不连续memref生成低效指令。
  * 语义合法性：实现reposition pass用于调整特定op的位置，使用use-def分析和SliceAnalysis依次完成wrap和clone。实现部分op的legalize pattern。
  * 支持新features：实现output alias，支持非dense value constant下降，支持i1类型的下降。
  * 完备性测试：修复多个算子下降流程，设计stablehlo、linalg算子测试方案并补充测试。

* ICT-TX跨平台编译联合项目 [2023.9-2024.3]
  * 优化slice、reduce等算子下降行为，将其转换为对底层指令更友好的IR。
  * 通过perf分析单算子的运行时间，为conv-like、pooling、element-wise等算子实现自适应 tile pattern，自适应获得不同shape下的最优tile策略。
  * 10万数据量下，DCNMix网络推理时间缩短到最初的1/30。累计完成该项目10次交付。

Skills
======
* 熟练使用CPP、Python，熟悉PyTorch编程
* 熟悉MLIR编程、了解LLVM、了解Triton编程与调优
* 熟悉常见推理优化技术和AI编译器相关的“调度与自动调优”技术


Certificates
======
* 校三好学生
* 赛艇国家二级运动员证书