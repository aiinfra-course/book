import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'AI Infra 技术系列',
  description: '深入理解 AI 推理系统原理与优化',
  base: '/',
  lang: 'zh-CN',

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '技术分享', link: '/talks/01-vllm-automatic-prefix-caching/' },
      { text: '02 · Continuous Batching', link: '/talks/02-vllm-continuous-batching/' },
      { text: '03 · V1 多进程架构', link: '/talks/03-vllm-core-engine/' },
      { text: '04 · LoRA 多适配器', link: '/talks/04-vllm-lora/' },
    ],

    sidebar: [
      {
        text: '技术分享',
        items: [
          {
            text: '01 · vLLM Automatic Prefix Caching',
            link: '/talks/01-vllm-automatic-prefix-caching/',
          },
          {
            text: '02 · vLLM Continuous Batching',
            link: '/talks/02-vllm-continuous-batching/',
          },
          {
            text: '03 · V1 多进程架构设计',
            link: '/talks/03-vllm-core-engine/',
          },
          {
            text: '04 · LoRA 多适配器支持',
            link: '/talks/04-vllm-lora/',
          },
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
