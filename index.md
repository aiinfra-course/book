---
layout: home

hero:
  name: "AI Infra"
  text: "技术系列"
  tagline: 深入理解 AI 推理系统的原理、源码与实践
  actions:
    - theme: brand
      text: 开始阅读
      link: /talks/01-vllm-automatic-prefix-caching/
    - theme: alt
      text: GitHub
      link: https://github.com/aiinfra-course/book

features:
  - icon: ⚡
    title: vLLM Automatic Prefix Caching
    details: 从零到熟悉 APC 机制：链式哈希、LRU 驱逐、BlockPool 数据结构，结合 vLLM v1 源码逐行分析。
    link: /talks/01-vllm-automatic-prefix-caching/
    linkText: 阅读文章

  - icon: 🔄
    title: vLLM Continuous Batching
    details: 深度解析连续批处理：Token 级调度、无 Prefill/Decode 分离、动态抢占与 KV Cache 管理。
    link: /talks/02-vllm-continuous-batching/
    linkText: 阅读文章

  - icon: 🚀
    title: V1 多进程架构设计
    details: 深入剖析 vLLM V1 的核心推理引擎架构：多进程设计、进程间通信、调度器与执行器协作。
    link: /talks/03-vllm-core-engine/
    linkText: 阅读文章

  - icon: 🎯
    title: LoRA 多适配器支持
    details: 理解 vLLM 的 LoRA 动态加载机制：多适配器管理、运行时配置、内存优化与性能提升。
    link: /talks/04-vllm-lora/
    linkText: 阅读文章

  - icon: 🔮
    title: Speculative Decoding
    details: 深入理解推测解码技术：Draft Model、EAGLE、Ngram 等多种方法的原理与实现。
    link: /talks/05-vllm-speculative-decoding/
    linkText: 阅读文章
---