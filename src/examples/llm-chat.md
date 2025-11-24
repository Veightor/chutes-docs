# LLM Chat Applications

This guide shows how to build powerful chat applications using Large Language Models (LLMs) with Chutes. We'll cover both high-performance VLLM serving and flexible SGLang implementations.

## Overview

Chutes provides pre-built templates for popular LLM serving frameworks:

- **VLLM**: High-performance serving with OpenAI-compatible APIs
- **SGLang**: Advanced serving with structured generation capabilities

Both frameworks support:

- Multi-GPU scaling for large models
- OpenAI-compatible endpoints
- Streaming responses
- Custom model configurations

## Quick Start: VLLM Chat Service

### Basic VLLM Setup

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

# Create a high-performance chat service
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",  # Required parameter
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    ),
    concurrency=4
)
```

### Production VLLM Configuration

For production workloads with larger models:

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

# Production-ready Mistral deployment
chute = build_vllm_chute(
    username="myuser",
    model_name="chutesai/Mistral-Small-3.1-24B-Instruct-2503",
    revision="cb765b56fbc11c61ac2a82ec777e3036964b975c",  # Required parameter moved to top level
    image="chutes/vllm:0.9.2.dev0",
    readme="Mistral-Small-3.1-24B-Instruct-2503 - High-performance chat model",
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=48,
        exclude=["l40", "a6000", "b200", "mi300x"],  # Exclude slower GPUs
    ),
    engine_args=dict(
        gpu_memory_utilization=0.97,
        max_model_len=96000,
        limit_mm_per_prompt={"image": 8},
        max_num_seqs=8,
        trust_remote_code=True,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tool_call_parser="mistral",
        enable_auto_tool_choice=True),
    concurrency=8)
```

## Advanced: SGLang with Custom Image

For more control and advanced features, use SGLang with a custom image:

```python
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
from chutes.image import Image

# Optimize networking for multi-GPU setups
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
for key in ["NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL"]:
    if key in os.environ:
        del os.environ[key]

# Build custom SGLang image with optimizations
image = (
    Image(
        username="myuser",
        name="sglang-optimized",
        tag="0.4.9.dev1",
        readme="SGLang with performance optimizations for large models")
    .from_base("parachutes/python:3.12")
    .run_command("pip install --upgrade pip")
    .run_command("pip install --upgrade 'sglang[all]'")
    .run_command(
        "git clone https://github.com/sgl-project/sglang sglang_src && "
        "cd sglang_src && pip install -e python[all]"
    )
    .run_command(
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu128 --upgrade"
    )
    .run_command("pip install datasets blobfile accelerate tiktoken")
    .run_command("pip install nvidia-nccl-cu12==2.27.6 --force-reinstall --no-deps")
    .with_env("SGL_ENABLE_JIT_DEEPGEMM", "1")
)

# Deploy Kimi K2 Instruct model
chute = build_sglang_chute(
    username="myuser",
    readme="Moonshot AI Kimi K2 Instruct - Advanced reasoning model",
    model_name="moonshotai/Kimi-K2-Instruct",
    image=image,
    concurrency=3,
    node_selector=NodeSelector(
        gpu_count=8,
        include=["h200"],  # Use latest H200 GPUs
    ),
    engine_args=(
        "--trust-remote-code "
        "--cuda-graph-max-bs 3 "
        "--mem-fraction-static 0.97 "
        "--context-length 65536 "
        "--revision d1e2b193ddeae7776463443e7a9aa3c3cdc51003 "
    ))
```

## Reasoning Models: DeepSeek R1

For advanced reasoning capabilities:

```python
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

# Deploy DeepSeek R1 reasoning model
chute = build_sglang_chute(
    username="myuser",
    readme="DeepSeek R1 - Advanced reasoning and problem-solving model",
    model_name="deepseek-ai/DeepSeek-R1",
    image="chutes/sglang:0.4.6.post5b",
    concurrency=24,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=140,  # Large memory requirement
        include=["h200"]),
    engine_args=(
        "--trust-remote-code "
        "--revision f7361cd9ff99396dbf6bd644ad846015e59ed4fc"
    ))
```

## Using Your Chat Service

### Deploy the Service

```bash
# Build and deploy your chat service
chutes deploy my_chat:chute

# Monitor deployment
chutes chutes get my-chat
```

### OpenAI-Compatible API

Both VLLM and SGLang provide OpenAI-compatible endpoints:

```bash
# Chat completions endpoint
curl -X POST "https://myuser-my-chat.chutes.ai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Streaming Responses

Enable real-time streaming for better user experience:

```bash
curl -X POST "https://myuser-my-chat.chutes.ai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Write a short story about AI"}
    ],
    "stream": true,
    "max_tokens": 500
  }'
```

### Python Client Example

```python
import openai

# Configure client to use your Chutes deployment
client = openai.OpenAI(
    base_url="https://myuser-my-chat.chutes.ai/v1",
    api_key="your-api-key"  # Or use environment variable
)

# Chat completion
response = client.chat.completions.create(
    model="microsoft/DialoGPT-medium",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)

# Streaming chat
stream = client.chat.completions.create(
    model="microsoft/DialoGPT-medium",
    messages=[
        {"role": "user", "content": "Tell me a joke"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Performance Optimization

### GPU Selection

Choose appropriate hardware for your model size:

```python
# For smaller models (7B-13B parameters)
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)

# For medium models (30B-70B parameters)
node_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=80
)

# For large models (100B+ parameters)
node_selector = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=140,
    include=["h200"]  # Use latest hardware
)
```

### Engine Optimization

Tune engine parameters for best performance:

```python
# VLLM optimizations
engine_args = dict(
    gpu_memory_utilization=0.97,  # Use most GPU memory
    max_model_len=32768,          # Context length
    max_num_seqs=16,              # Batch size
    trust_remote_code=True,       # Enable custom models
    enforce_eager=False,          # Use CUDA graphs
    disable_log_requests=True,    # Reduce logging overhead
)

# SGLang optimizations
engine_args = (
    "--trust-remote-code "
    "--cuda-graph-max-bs 8 "      # CUDA graph batch size
    "--mem-fraction-static 0.95 " # Memory allocation
    "--context-length 32768 "     # Context window
)
```

### Concurrency Settings

Balance throughput and resource usage:

```python
# High throughput setup
chute = build_vllm_chute(
    # ... other parameters
    concurrency=16,  # Handle many concurrent requests
    engine_args=dict(
        max_num_seqs=32,         # Large batch size
        gpu_memory_utilization=0.90)
)

# Low latency setup
chute = build_vllm_chute(
    # ... other parameters
    concurrency=4,   # Fewer concurrent requests
    engine_args=dict(
        max_num_seqs=8,          # Smaller batch size
        gpu_memory_utilization=0.95)
)
```

## Monitoring and Troubleshooting

### Check Service Status

```bash
# View service health
chutes chutes get my-chat

# View recent logs
chutes chutes logs my-chat

# Monitor resource usage
chutes chutes metrics my-chat
```

### Common Issues

**Out of Memory (OOM)**

```python
# Reduce memory usage
engine_args = dict(
    gpu_memory_utilization=0.85,  # Lower memory usage
    max_model_len=16384,          # Shorter context
    max_num_seqs=4,               # Smaller batch
)
```

**Slow Response Times**

```python
# Optimize for speed
engine_args = dict(
    enforce_eager=False,          # Enable CUDA graphs
    disable_log_requests=True,    # Reduce logging
    quantization="awq",           # Use quantization
)
```

**Connection Timeouts**

```python
# Increase timeouts
chute = build_vllm_chute(
    # ... other parameters
    concurrency=8,  # Increase concurrent capacity
    engine_args=dict(
        max_num_seqs=16,  # Larger batches
    )
)
```

## Best Practices

### 1. Model Selection

- **For general chat**: Mistral, Llama, or Qwen models
- **For reasoning**: DeepSeek R1, GPT-4 style models
- **For coding**: CodeLlama, DeepSeek Coder
- **For multilingual**: Qwen, multilingual Mistral variants

### 2. Resource Planning

- Start with smaller configurations and scale up
- Monitor GPU utilization and adjust concurrency
- Use appropriate GPU types for your model size
- Consider cost vs. performance trade-offs

### 3. Development Workflow

```bash
# 1. Test locally with small model
chutes deploy test-chat:chute --wait

# 2. Validate API endpoints
curl https://myuser-test-chat.chutes.ai/v1/models

# 3. Load test with production model
chutes deploy prod-chat:chute --wait

# 4. Monitor and optimize
chutes chutes metrics prod-chat
```

### 4. Security Considerations

- Use API keys for authentication
- Implement rate limiting if needed
- Monitor usage and costs
- Keep model revisions pinned for reproducibility

## Next Steps

- **Advanced Features**: Explore function calling and tool use
- **Custom Templates**: Build specialized chat applications
- **Integration**: Connect with web frontends and mobile apps
- **Scaling**: Implement load balancing across multiple deployments

For more examples, see:

- [Streaming Responses](/docs/examples/streaming-responses)
- [Custom Images](/docs/examples/custom-images)
- [Templates Documentation](/docs/templates/)
