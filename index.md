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
---
