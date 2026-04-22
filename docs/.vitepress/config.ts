import { defineConfig } from "vitepress";

const GITHUB_URL = "https://github.com/zlynx-ai/zlynx"

export default defineConfig({
  title: "Zlynx",
  description: "Experimental JAX and Flax NNX deep learning library documentation.",
  lastUpdated: true,
  cleanUrls: true,
  themeConfig: {
    nav: [
      // { text: "Home", link: "/" },
      { text: "Tutorials", link: "/tutorials/mnist" },
      { text: "API Reference", link: "/api-references/api-references" },
      { text: "GitHub", link: GITHUB_URL }
    ],
    sidebar: [
      {
        text: "Getting started",
        items: [
          { text: "Quick start", link: "/getting-started/quick-start" },
          { text: "Installation", link: "/getting-started/installation" },
          { text: "Create a model", link: "/getting-started/create-a-model" },
          { text: "Training", link: "/getting-started/training" },
          { text: "Checkpoint", link: "/getting-started/ckpt" },
          { text: "Sharding", link: "/getting-started/sharding" },
          { text: "PEFT", link: "/getting-started/peft" },
          { text: "GaLore", link: "/getting-started/galore" }
        ]
      },
      {
        text: "Examples",
        items: [
          { text: "MNIST", link: "/examples/mnist" },
        ]
      },
      {
        text: "API Reference",
        items: [
          { text: "API Index", link: "/api-references/api-references" },
          { text: "Models", link: "/api-references/models" },
          { text: "Modules", link: "/api-references/modules" },
          { text: "Trainer", link: "/api-references/trainer" }
        ]
      }
    ],
    socialLinks: [
      { icon: "github", link: GITHUB_URL }
    ],
    footer: {
      message: "Experimental project. APIs and behavior may change.",
      copyright: "Copyright © 2026 Shinapri"
    }
  }
});
