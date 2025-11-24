# VLLM Template

The **VLLM template** is the most popular way to deploy large language models on Chutes. It provides a high-performance, OpenAI-compatible API server powered by [vLLM](https://docs.vllm.ai/), optimized for fast inference and high throughput.

## What is VLLM?

VLLM is a fast and memory-efficient inference engine for large language models that provides:

- 📈 **High throughput** serving with PagedAttention
- 🧠 **Memory efficiency** with optimized attention algorithms
- 🔄 **Continuous batching** for better GPU utilization
- 🌐 **OpenAI-compatible API** for easy integration
- ⚡ **Multi-GPU support** for large models

## Quick Start

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",  # Required: locks model to specific version
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    )
)
```

That's it! This creates a complete VLLM deployment with:

- ✅ Automatic model downloading and caching
- ✅ OpenAI-compatible `/v1/chat/completions` endpoint
- ✅ Built-in streaming support
- ✅ Optimized inference settings
- ✅ Auto-scaling based on demand

## Function Reference

### `build_vllm_chute()`

```python
def build_vllm_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    revision: str,
    image: str | Image = VLLM,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 32,
    engine_args: Dict[str, Any] = {}) -> VLLMChute
```

#### Required Parameters

**`username: str`**  
Your Chutes username.

**`model_name: str`**  
HuggingFace model identifier (e.g., `"microsoft/DialoGPT-medium"`).

**`node_selector: NodeSelector`**  
Hardware requirements specification.

**`revision: str`**  
**Required.** Git revision/commit hash to lock the model version. Use the current `main` branch commit for reproducible deployments.

```python
# Get current revision from HuggingFace
revision = "cb765b56fbc11c61ac2a82ec777e3036964b975c"
```

#### Optional Parameters

**`image: str | Image = VLLM`**  
Docker image to use. Defaults to the official Chutes VLLM image.

**`tagline: str = ""`**  
Short description for your chute.

**`readme: str = ""`**  
Markdown documentation for your chute.

**`concurrency: int = 32`**  
Maximum concurrent requests per instance.

**`engine_args: Dict[str, Any] = {}`**  
VLLM engine configuration options. See [Engine Arguments](#engine-arguments).

## Engine Arguments

The `engine_args` parameter allows you to configure VLLM's behavior:

### Memory and Performance

```python
engine_args = {
    # Memory utilization (0.0-1.0)
    "gpu_memory_utilization": 0.95,

    # Maximum sequence length
    "max_model_len": 4096,

    # Maximum number of sequences to process in parallel
    "max_num_seqs": 256,

    # Enable chunked prefill for long sequences
    "enable_chunked_prefill": True,

    # Maximum number of tokens in a single chunk
    "max_num_batched_tokens": 8192,
}
```

### Model Loading

```python
engine_args = {
    # Tensor parallelism (automatically set based on GPU count)
    "tensor_parallel_size": 2,

    # Pipeline parallelism
    "pipeline_parallel_size": 1,

    # Data type for model weights
    "dtype": "auto",  # or "float16", "bfloat16", "float32"

    # Quantization method
    "quantization": "awq",  # or "gptq", "squeezellm", etc.

    # Trust remote code (for custom models)
    "trust_remote_code": True,
}
```

### Advanced Features

```python
engine_args = {
    # Enable prefix caching
    "enable_prefix_caching": True,

    # Speculative decoding
    "speculative_model": "microsoft/DialoGPT-small",
    "num_speculative_tokens": 5,

    # Guided generation
    "guided_decoding_backend": "outlines",

    # Disable logging for better performance
    "disable_log_stats": True,
    "disable_log_requests": True,
}
```

## Hardware Configuration

### GPU Requirements

Choose hardware based on your model size:

#### Small Models (< 7B parameters)

```python
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["l40", "a6000", "a100"]
)
```

#### Medium Models (7B - 13B parameters)

```python
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["a100", "h100"]
)
```

#### Large Models (13B - 70B parameters)

```python
node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40,
    include=["a100", "h100"]
)
```

#### Huge Models (70B+ parameters)

```python
node_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=80,
    include=["h100"]
)
```

### GPU Type Selection

**High Performance:**

```python
include=["h100", "a100"]  # Latest, fastest GPUs
```

**Balanced:**

```python
include=["a100", "l40", "a6000"]  # Good performance/cost ratio
```

**Budget:**

```python
exclude=["h100"]  # Exclude most expensive GPUs
```

## API Endpoints

The VLLM template provides OpenAI-compatible endpoints:

### Chat Completions

**POST `/v1/chat/completions`**

```python
import aiohttp

async def chat_completion():
    url = "https://myuser-mychute.chutes.ai/v1/chat/completions"

    payload = {
        "model": "microsoft/DialoGPT-medium",
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(result["choices"][0]["message"]["content"])
```

### Streaming Chat

```python
async def streaming_chat():
    url = "https://myuser-mychute.chutes.ai/v1/chat/completions"

    payload = {
        "model": "microsoft/DialoGPT-medium",
        "messages": [
            {"role": "user", "content": "Tell me a story"}
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="")
```

### Text Completions

**POST `/v1/completions`**

```python
payload = {
    "model": "microsoft/DialoGPT-medium",
    "prompt": "The future of AI is",
    "max_tokens": 50,
    "temperature": 0.7
}
```

### Tokenization

**POST `/tokenize`**

```python
payload = {
    "model": "microsoft/DialoGPT-medium",
    "text": "Hello, world!"
}
# Returns: {"tokens": [1, 2, 3, ...]}
```

**POST `/detokenize`**

```python
payload = {
    "model": "microsoft/DialoGPT-medium",
    "tokens": [1, 2, 3]
}
# Returns: {"text": "Hello, world!"}
```

## Complete Examples

### Basic Chat Model

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    tagline="Conversational AI chatbot",
    readme="""
    # My Chat Bot

    A conversational AI powered by DialoGPT.

    ## Usage
    Send POST requests to `/v1/chat/completions` with your messages.
    """,
    concurrency=16
)
```

### High-Performance Large Model

```python
chute = build_vllm_chute(
    username="myuser",
    model_name="meta-llama/Llama-2-70b-chat-hf",
    revision="latest-commit-hash",
    node_selector=NodeSelector(
        gpu_count=4,
        min_vram_gb_per_gpu=80,
        include=["h100", "a100"]
    ),
    engine_args={
        "gpu_memory_utilization": 0.95,
        "max_model_len": 4096,
        "max_num_seqs": 128,
        "enable_chunked_prefill": True,
        "trust_remote_code": True,
    },
    concurrency=64
)
```

### Code Generation Model

```python
chute = build_vllm_chute(
    username="myuser",
    model_name="Phind/Phind-CodeLlama-34B-v2",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=40
    ),
    engine_args={
        "max_model_len": 8192,  # Longer context for code
        "temperature": 0.1,     # More deterministic for code
    },
    tagline="Advanced code generation AI"
)
```

### Quantized Model for Efficiency

```python
chute = build_vllm_chute(
    username="myuser",
    model_name="TheBloke/Llama-2-13B-chat-AWQ",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16  # Much less VRAM needed
    ),
    engine_args={
        "quantization": "awq",
        "gpu_memory_utilization": 0.9,
    }
)
```

## Testing Your Deployment

### Local Testing

Before deploying, test your configuration:

```python
# Add to your chute file
if __name__ == "__main__":
    import asyncio

    async def test():
        response = await chute.chat({
            "model": "your-model-name",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        })
        print(response)

    asyncio.run(test())
```

Run locally:

```bash
chutes run my_vllm_chute:chute --dev
```

### Production Testing

After deployment:

```bash
curl -X POST https://myuser-mychute.chutes.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [{"role": "user", "content": "Test message"}],
    "max_tokens": 50
  }'
```

## Performance Optimization

### Memory Optimization

```python
engine_args = {
    # Use maximum available memory
    "gpu_memory_utilization": 0.95,

    # Enable memory-efficient attention
    "enable_chunked_prefill": True,

    # Optimize for your typical sequence length
    "max_model_len": 2048,  # Adjust based on your use case
}
```

### Throughput Optimization

```python
engine_args = {
    # Increase parallel sequences
    "max_num_seqs": 512,

    # Larger batch sizes
    "max_num_batched_tokens": 16384,

    # Disable logging in production
    "disable_log_stats": True,
    "disable_log_requests": True,
}
```

### Latency Optimization

```python
engine_args = {
    # Smaller batch sizes for lower latency
    "max_num_seqs": 32,

    # Enable prefix caching
    "enable_prefix_caching": True,

    # Use speculative decoding for faster generation
    "speculative_model": "smaller-model-name",
    "num_speculative_tokens": 5,
}
```

## Troubleshooting

### Common Issues

**Out of Memory Errors**

```python
# Reduce memory usage
engine_args = {
    "gpu_memory_utilization": 0.8,  # Lower from 0.95
    "max_model_len": 2048,           # Reduce max length
    "max_num_seqs": 64,              # Fewer parallel sequences
}
```

**Slow Model Loading**

```python
# The model downloads on first startup
# Check logs: chutes chutes get your-chute
# Subsequent starts are fast due to caching
```

**Model Not Found**

```python
# Ensure model exists and is public
# Check: https://huggingface.co/microsoft/DialoGPT-medium
# Use exact model name from HuggingFace
```

**Deployment Fails**

```bash
# Check image build status
chutes images list --name your-image

# Verify configuration
python -c "from my_chute import chute; print(chute.node_selector)"
```

### Performance Issues

**Low Throughput**

- Increase `max_num_seqs` and `max_num_batched_tokens`
- Use more GPUs with `tensor_parallel_size`
- Enable `enable_chunked_prefill`

**High Latency**

- Reduce `max_num_seqs` for lower batching
- Enable `enable_prefix_caching`
- Use faster GPU types (H100 > A100 > L40)

**Memory Issues**

- Lower `gpu_memory_utilization`
- Reduce `max_model_len`
- Consider quantized models (AWQ, GPTQ)

## Best Practices

### 1. Model Selection

- Use quantized models (AWQ/GPTQ) for better efficiency
- Choose the smallest model that meets your quality requirements
- Test with different model variants

### 2. Hardware Sizing

- Start with minimum requirements and scale up
- Monitor GPU utilization in the dashboard
- Use `include`/`exclude` filters for cost optimization

### 3. Performance Tuning

- Set `revision` to lock model versions
- Tune `engine_args` for your specific use case
- Enable logging initially, disable in production

### 4. Monitoring

- Check the Chutes dashboard for metrics
- Monitor request latency and throughput
- Set up alerts for failures

## Advanced Features

### Custom Chat Templates

```python
engine_args = {
    "chat_template": """
    {%- for message in messages %}
        {%- if message['role'] == 'user' %}
            Human: {{ message['content'] }}
        {%- elif message['role'] == 'assistant' %}
            Assistant: {{ message['content'] }}
        {%- endif %}
    {%- endfor %}
    Assistant:
    """
}
```

### Tool Calling

```python
engine_args = {
    "tool_call_parser": "mistral",
    "enable_auto_tool_choice": True,
}
```

### Guided Generation

```python
engine_args = {
    "guided_decoding_backend": "outlines",
}

# Then in your requests:
{
    "guided_json": {"type": "object", "properties": {"name": {"type": "string"}}}
}
```

## Migration from Other Platforms

### From OpenAI

Replace the base URL and use your model name:

```python
# Before (OpenAI)
client = OpenAI(api_key="sk-...")

# After (Chutes)
client = OpenAI(
    api_key="dummy",  # Not needed for Chutes
    base_url="https://myuser-mychute.chutes.ai/v1"
)
```

### From Hugging Face Transformers

VLLM is much faster than transformers for serving:

```python
# Before (Transformers)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model-name")

# After (Chutes VLLM)
chute = build_vllm_chute(
    username="myuser",
    model_name="model-name",
    # ... configuration
)
```

## Next Steps

- **[SGLang Template](/docs/templates/sglang)** - Alternative high-performance LLM serving
- **[Custom Images](/docs/guides/custom-images)** - Build your own VLLM images
- **[Streaming Guide](/docs/guides/streaming)** - Advanced streaming patterns
- **[Examples](/docs/examples/llm-chat)** - Complete application examples
