# Quick Start Guide

Get your first chute deployed in under 10 minutes! This guide will walk you through creating, building, and deploying a simple AI application.

## Prerequisites

Make sure you've completed the [Installation & Setup](installation) guide first.

## Step 1: Create Your First Chute

Let's build a simple text generation chute using a pre-built template.

Create a new file called `my_first_chute.py`:

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

# Build a chute using the VLLM template
chute = build_vllm_chute(
    username="your-username",  # Replace with your Chutes username
    model_name="microsoft/DialoGPT-medium",
    revision="main",  # Required: locks model to specific version
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    concurrency=4,
    readme="""
    # My First Chute
    A simple conversational AI powered by DialoGPT.

    ## Usage
    Send a POST request to `/v1/chat/completions` with your message.
    """
)
```

That's it! You've just defined a complete AI application with:

- ✅ A pre-configured VLLM server
- ✅ Automatic model downloading
- ✅ OpenAI-compatible API endpoints
- ✅ GPU resource requirements
- ✅ Auto-scaling configuration

## Step 2: Build Your Image

Build the Docker image for your chute:

```bash
chutes build my_first_chute:chute --wait
```

This will:

- 📦 Create a Docker image with all dependencies
- 🔧 Install VLLM and required libraries
- ⬇️ Pre-download your model
- ✅ Validate the configuration

The `--wait` flag streams the build logs to your terminal so you can monitor progress.

## Step 3: Deploy Your Chute

Deploy your chute to the Chutes platform:

```bash
chutes deploy my_first_chute:chute
```

After deployment, you'll see output like:

```
✅ Chute deployed successfully!
🌐 Public API: https://your-username-my-first-chute.chutes.ai
📋 Chute ID: 12345678-1234-5678-9abc-123456789012
```

## Step 4: Test Your Chute

Your chute is now live! Test it with a simple chat completion:

### Option 1: Using curl

```bash
curl -X POST https://your-username-my-first-chute.chutes.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello! How are you today?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Option 2: Using Python

```python
import asyncio
import aiohttp
import json

async def chat_with_chute():
    url = "https://your-username-my-first-chute.chutes.ai/v1/chat/completions"

    payload = {
        "model": "microsoft/DialoGPT-medium",
        "messages": [
            {"role": "user", "content": "Hello! How are you today?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(json.dumps(result, indent=2))

# Run the test
asyncio.run(chat_with_chute())
```

### Option 3: Test Locally

You can also test your chute locally before deploying using the CLI:

```bash
# Run your chute locally
chutes run my_first_chute:chute --dev

# Then in another terminal, test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello! How are you today?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Step 5: Monitor and Manage

### View Your Chutes

```bash
chutes chutes list
```

### Get Detailed Information

```bash
chutes chutes get my-first-chute
```

### Check Logs

Visit the [Chutes Dashboard](https://chutes.ai) to view real-time logs and metrics.

## What Just Happened?

Congratulations! You just:

1. 🎯 **Defined** an AI application with just a few lines of Python
2. 🏗️ **Built** a production-ready Docker image
3. 🚀 **Deployed** to GPU-accelerated infrastructure
4. 🌐 **Exposed** OpenAI-compatible API endpoints
5. 💰 **Pay-per-use** - only charged when your chute receives requests

## Next Steps

Now that you have a working chute, explore more advanced features:

### 🎨 Try Different Models

Replace `microsoft/DialoGPT-medium` with:

- `microsoft/DialoGPT-large` (requires more VRAM)
- `facebook/blenderbot-400M-distill`
- `google/flan-t5-base`

### 🔧 Customize Hardware

Adjust your `NodeSelector`:

```python
NodeSelector(
    gpu_count=2,           # Use 2 GPUs
    min_vram_gb_per_gpu=24, # Require 24GB VRAM per GPU
    include=["a100", "h100"], # Prefer specific GPU types
    exclude=["k80"]        # Avoid older GPUs
)
```

### 🎛️ Tune Performance

Modify engine arguments:

```python
chute = build_vllm_chute(
    # ... other parameters ...
    engine_args={
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 32
    }
)
```

### 📚 Learn Core Concepts

- **[Understanding Chutes](../core-concepts/chutes)** - Deep dive into the Chute class
- **[Cords (API Endpoints)](../core-concepts/cords)** - Custom API endpoints
- **[Custom Images](../core-concepts/images)** - Build your own Docker images

### 🏗️ Build Custom Applications

- **[Your First Custom Chute](first-chute)** - Build from scratch
- **[Custom Image Building](../guides/custom-images)** - Advanced Docker setups
- **[Input/Output Schemas](../guides/schemas)** - Type-safe APIs

## Common Questions

**Q: How much does this cost?**
A: You only pay for GPU time when your chute is processing requests. Idle time is free!

**Q: Can I use my own models?**
A: Yes! Upload models to HuggingFace or use the custom image building features.

**Q: What about scaling?**
A: Chutes automatically scales based on demand. Configure `concurrency` to control how many requests each instance handles.

**Q: How do I debug issues?**
A: Check the logs in the [Chutes Dashboard](https://chutes.ai) or use the CLI: `chutes chutes get my-chute`

## Troubleshooting

**Build failed?**

- Check that your model name is correct
- Ensure you have sufficient developer deposit
- Try with a smaller model first

**Deployment failed?**

- Verify your image built successfully
- Check your username and chute name are valid
- Ensure you have proper permissions

**Can't access your chute?**

- Wait a few minutes for DNS propagation
- Check the exact URL from `chutes chutes get`
- Verify the chute is in "running" status

## Get Help

- 📖 **Detailed Guides**: Continue with [Your First Custom Chute](/docs/getting-started/first-chute)
- 💬 **Community**: [Join our Discord](https://discord.gg/wHrXwWkCRz)
- 🐛 **Issues**: [GitHub Issues](https://github.com/rayonlabs/chutes/issues)
- 📧 **Support**: `support@chutes.ai`

---

Ready to build something more advanced? Check out [Your First Custom Chute](/docs/getting-started/first-chute) to learn how to build completely custom applications!
