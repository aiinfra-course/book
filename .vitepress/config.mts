import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'AI Infra 技术系列',
  description: '深入理解 AI 推理系统原理与优化',
  base: '/book/',
  lang: 'zh-CN',

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '01 · APC', link: '/talks/01-vllm-automatic-prefix-caching/' },
      { text: '02 · Continuous Batching', link: '/talks/02-vllm-continuous-batching/' },
      { text: '03 · V1 架构', link: '/talks/03-vllm-core-engine/' },
      { text: '04 · LoRA', link: '/talks/04-vllm-lora/' },
      { text: '05 · Speculative Decoding', link: '/talks/05-vllm-speculative-decoding/' },
      { text: '06 · Multimodal', link: '/talks/06-vllm-multimodal/' },
      { text: '07 · Quantization', link: '/talks/07-vllm-quantization/' },
      { text: '08 · CUDA Graphs', link: '/talks/08-vllm-cuda-graphs/' },
      { text: '09 · Parallel', link: '/talks/09-vllm-parallel/' },
      { text: '10 · Function Calling', link: '/talks/10-vllm-function-calling/' },
      { text: '11 · Structured Output', link: '/talks/11-vllm-structured-output/' },
      { text: '12 · Sampling', link: '/talks/12-vllm-sampling/' },
      { text: '13 · Monitoring', link: '/talks/13-vllm-monitoring/' },
      { text: '14 · Platforms', link: '/talks/14-vllm-platforms/' },
    ],

    sidebar: [
      {
        text: '技术分享',
        items: [
          { text: '01 · vLLM Automatic Prefix Caching', link: '/talks/01-vllm-automatic-prefix-caching/' },
          { text: '02 · vLLM Continuous Batching', link: '/talks/02-vllm-continuous-batching/' },
          { text: '03 · V1 多进程架构设计', link: '/talks/03-vllm-core-engine/' },
          { text: '04 · LoRA 多适配器支持', link: '/talks/04-vllm-lora/' },
          { text: '05 · Speculative Decoding', link: '/talks/05-vllm-speculative-decoding/' },
          { text: '06 · 多模态输入处理', link: '/talks/06-vllm-multimodal/' },
          { text: '07 · 量化策略与压缩技术', link: '/talks/07-vllm-quantization/' },
          { text: '08 · 编译优化与 CUDA Graphs', link: '/talks/08-vllm-cuda-graphs/' },
          { text: '09 · 分布式并行策略', link: '/talks/09-vllm-parallel/' },
          { text: '10 · 工具调用与 Function Calling', link: '/talks/10-vllm-function-calling/' },
          { text: '11 · 结构化输出与 JSON Schema', link: '/talks/11-vllm-structured-output/' },
          { text: '12 · 高级采样与生成策略', link: '/talks/12-vllm-sampling/' },
          { text: '13 · 监控与可观测性', link: '/talks/13-vllm-monitoring/' },
          { text: '14 · 多平台支持', link: '/talks/14-vllm-platforms/' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/aiinfra-course/book' },
    ],

    footer: {
      message: 'AI Infra 技术系列',
      copyright: 'Copyright © 2024 aiinfra-course',
    },

    outline: {
      label: '本页目录',
      level: [2, 3],
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '深色模式',
  },

  markdown: {
    lineNumbers: true,
  },
}))
