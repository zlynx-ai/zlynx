# Zlynx Documentation

> [!CAUTION]
> **Zlynx is currently an experimental library.** APIs are subject to change without notice. We are in the early stages of development and welcome feedback.

An experimental, research-oriented deep learning library built on **JAX** and **Flax NNX**. Zlynx explores providing fine-grained control over model architectures, training loops, and distributed setups without the weight of larger frameworks.

---

## 🚀 Getting Started

If you are exploring Zlynx, these guides offer an early look at setting up and running initial experiments.

- **[Installation](./getting-started/installation.md)** — Basic setup for Zlynx and its dependencies.
- **[Quick Start](./getting-started/quick-start.md)** — A brief overview of current experimental workflows.
- **[Create a Model](./getting-started/create-a-model.md)** — Defining architectures using the experimental `Z` base class.
- **[Training](./getting-started/training.md)** — An introduction to the current `Trainer` implementation.
- **[Checkpointing](./getting-started/ckpt.md)** — Early support for saving and loading models.

## 💡 Concepts & Explorations

Deep dives into the experimental interfaces and ideas currently in Zlynx.

- **[Sharding & Parallelism](./useful-stuff/sharding.md)** — Initial concepts for distributed training.
- **[PEFT (LoRA, etc.)](./useful-stuff/peft.md)** — Experimental parameter-efficient fine-tuning utilities.
- **[GaLore](./useful-stuff/galore.md)** — Exploratory memory-efficient full fine-tuning.
- **[Logging Backends](./useful-stuff/logging-backend.md)** — Current state of logging integrations.

## 📚 Examples

See how Zlynx can be used in its current state.

- **[MNIST Tutorial](./examples/mnist.md)** — A basic training example using Zlynx.

## 🛠️ API Reference (WIP)

Documentation for the available parts of the Zlynx project.

- **[Core API](./api/core.md)** — Base classes and initial utilities.
- **[Model API](./api/model.md)** — Built-in architectures (Under Development).
- **[Module API](./api/module.md)** — Current building blocks: Attention, MLP, RoPE, and PEFT.
- **[Trainer API](./api/trainer.md)** — The experimental training loop and configuration.

---

### Project Goals

- **Exploratory** — Built for research and experimentation with JAX/NNX.
- **Modularity** — Exploring reusable blocks for custom architectures.
- **Scalability** — Aims to simplify sharding and parallelism in the long term.
- **Integration** — Initial efforts to support industry standards like SafeTensors and HuggingFace Hub.
