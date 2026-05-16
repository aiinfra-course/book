import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI Infra 技术系列',
  description: '深入理解 AI 推理系统原理与优化',
  base: '/book/',
  lang: 'zh-CN',

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '技术分享', link: '/talks/01-vllm-automatic-prefix-caching/' },
      { text: '02 · Continuous Batching', link: '/talks/02-vllm-continuous-batching/' },
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
            items: [
              { text: '完整文档', link: '/talks/02-vllm-continuous-batching/' },
            ],
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
})
