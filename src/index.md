# Chutes SDK Documentation

Welcome to the complete documentation for the **Chutes SDK** - a powerful Python framework for building and deploying serverless AI applications on GPU-accelerated infrastructure.

## What is Chutes?

Chutes is a serverless AI compute platform that allows you to:

- 🚀 Deploy AI models and applications instantly
- 💰 Pay only for GPU time you actually use
- 🔧 Build custom Docker images or use pre-built templates
- 📊 Scale automatically based on demand
- 🎯 Focus on your AI logic, not infrastructure management

## Quick Start

```bash
# Install the Chutes SDK
pip install chutes

# Register your account
chutes register

# Deploy your first chute
chutes deploy my_chute:chute
```

## Key Features

### 🎯 **Simple Decorator-Based API**

Define your AI endpoints with simple Python decorators:

```python
@chute.cord(public_api_path="/generate")
async def generate_text(self, prompt: str) -> str:
    return await self.model.generate(prompt)
```

### 🔧 **Flexible Templates**

Get started quickly with pre-built templates for popular AI frameworks:

```python
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(gpu_count=1)
)
```

### 🏗️ **Custom Image Building**

Build sophisticated Docker environments with a fluent API:

```python
image = (
    Image(username="myuser", name="custom-ai", tag="1.0")
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch transformers")
    .with_env("MODEL_PATH", "/app/models")
)
```

### ⚡ **Hardware Optimization**

Specify exactly the hardware you need:

```python
node_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=80,
    exclude=["old_gpus"]
)
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Your Code     │    │   Chutes SDK    │    │ Chutes Platform │
│                 │    │                 │    │                 │
│ @chute.cord     │───▶│ Build & Deploy  │───▶│ GPU Clusters    │
│ def generate()  │    │                 │    │                 │
│                 │    │ HTTP APIs       │    │ Auto-scaling    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Community & Support

- 📖 **Documentation**: You're here!
- 💬 **Discord**: [Join our community](https://discord.gg/wHrXwWkCRz)
- 🐛 **Issues**: [GitHub Issues](https://github.com/rayonlabs/chutes)

---

Ready to get started? Head to the [Installation Guide](getting-started/installation) to begin your Chutes journey!
