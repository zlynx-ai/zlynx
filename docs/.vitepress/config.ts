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
      { text: "API Reference", link: "/api/" },
      { text: "GitHub", link: GITHUB_URL }
    ],
    sidebar: [
      {
        text: "Getting Started",
        link: "/getting-started/",
        items: [
          { text: "Quick start", link: "/getting-started/quick-start" },
          { text: "Installation", link: "/getting-started/installation" },
          { text: "Create a model", link: "/getting-started/create-a-model" },
          { text: "Training", link: "/getting-started/training" },
          { text: "Checkpoint", link: "/getting-started/ckpt" },
        ]
      },
      {
        text: "Useful Stuff",
        link: "/useful-stuff/",
        items: [
          { text: "Sharding", link: "/useful-stuff/sharding" },
          { text: "PEFT", link: "/useful-stuff/peft" },
          { text: "GaLore", link: "/useful-stuff/galore" },
          { text: "Logging backend", link: "/useful-stuff/logging-backend" }
        ]
      },
      {
        text: "Examples",
        link: "/examples/",
        items: [
          { text: "MNIST", link: "/examples/mnist" },
        ]
      },
      {
        text: "API Reference",
        link: "/api/",
        items: [
          {
            text: "Core",
            link: "/api/core",
            collapsed: true,
            items: [
              { text: "Z", link: "/api/core-z" },
              { text: "Configs and Outputs", link: "/api/core-configs-outputs" },
              { text: "Inference", link: "/api/core-inference" },
            ]
          },
          {
            text: "Model",
            link: "/api/model",
            collapsed: true,
            items: [
              { text: "Llama", link: "/api/model-llama" },
              { text: "DiT", link: "/api/model-dit" },
              { text: "SSM", link: "/api/model-ssm" },
              {
                text: "OpenAI",
                link: "/api/model-openai",
                collapsed: true,
                items: [
                  { text: "gpt-oss", link: "/api/model-openai-gptoss" },
                ]
              },
              {
                text: "Google",
                link: "/api/model-google",
                collapsed: true,
                items: [
                  { text: "SigLIP", link: "/api/model-google-siglip" },
                  { text: "SigLIP 2", link: "/api/model-google-siglip2" },
                  { text: "PaliGemma", link: "/api/model-google-paligemma" },
                ]
              },
              {
                text: "DeepMind",
                link: "/api/model-deepmind",
                collapsed: true,
                items: [
                  { text: "Gemma 2", link: "/api/model-deepmind-gemma2" },
                  { text: "Gemma 3n", link: "/api/model-deepmind-gemma3n" },
                  { text: "Gemma 4", link: "/api/model-deepmind-gemma4" },
                ]
              },
              {
                text: "DeepSeek",
                link: "/api/model-deepseek",
                collapsed: true,
                items: [
                  { text: "DeepSeek-V3", link: "/api/model-deepseek-v3" },
                  { text: "DeepSeek-R1", link: "/api/model-deepseek-r1" },
                ]
              },
              {
                text: "Alibaba",
                link: "/api/model-alibaba",
                collapsed: true,
                items: [
                  { text: "Qwen", link: "/api/model-alibaba-qwen" },
                ]
              },
              {
                text: "Moonshot",
                link: "/api/model-moonshot",
                collapsed: true,
                items: [
                  { text: "Kimi", link: "/api/model-moonshot-kimi" },
                ]
              },
              {
                text: "Zhipu",
                link: "/api/model-zhipu",
                collapsed: true,
                items: [
                  { text: "GLM", link: "/api/model-zhipu-glm" },
                ]
              },
              {
                text: "NVIDIA",
                link: "/api/model-nvidia",
                collapsed: true,
                items: [
                  { text: "PersonaPlex", link: "/api/model-nvidia-personaplex" },
                ]
              },
              {
                text: "Black Forest Labs",
                link: "/api/model-blackforestlabs",
                collapsed: true,
                items: [
                  { text: "FLUX", link: "/api/model-blackforestlabs-flux" },
                ]
              },
              {
                text: "Stability AI",
                link: "/api/model-stabilityai",
                collapsed: true,
                items: [
                  { text: "Stable Diffusion", link: "/api/model-stabilityai-stable-diffusion" },
                ]
              },
            ]
          },
          {
            text: "Module",
            link: "/api/module",
            collapsed: true,
            items: [
              { text: "Layers", link: "/api/module-layers" },
              { text: "PEFT", link: "/api/module-peft" },
            ]
          },
          {
            text: "Trainer",
            link: "/api/trainer",
            collapsed: true,
            items: [
              { text: "TrainerConfig", link: "/api/trainer-config" },
              { text: "Optimizer and Schedulers", link: "/api/trainer-optim" },
            ]
          }
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
