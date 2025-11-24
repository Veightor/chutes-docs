# Templates API Reference

Chutes provides pre-built templates for common AI/ML frameworks and use cases. This reference covers all available template functions, their parameters, and customization options.

## Overview

Templates in Chutes are factory functions that create pre-configured `Chute` instances with optimized settings for specific AI frameworks. They provide:

- **Quick Setup**: Instant deployment of popular AI models
- **Best Practices**: Pre-configured optimization settings
- **Consistent Configuration**: Standardized deployment patterns
- **Customization**: Full control over parameters and settings

## Available Templates

### VLLM Templates

#### `build_vllm_chute()`

Create a chute optimized for VLLM (high-performance LLM serving).

**Signature:**

```python
def build_vllm_chute(
    username: str,
    name: Optional[str] = None,
    readme: Optional[str] = None,
    model_name: str,
    image: Optional[str] = None,
    node_selector: Optional[NodeSelector] = None,
    engine_args: Optional[Dict[str, Any]] = None,
    concurrency: int = 1,
    **kwargs
) -> Chute
```

**Parameters:**

- **`username: str`** - Your Chutes username (required)
- **`name: Optional[str]`** - Chute name (defaults to model name)
- **`readme: Optional[str]`** - Custom documentation
- **`model_name: str`** - HuggingFace model identifier (required)
- **`image: Optional[str]`** - Custom VLLM image (defaults to latest)
- **`node_selector: Optional[NodeSelector]`** - Hardware requirements
- **`engine_args: Optional[Dict[str, Any]]`** - VLLM engine configuration
- **`concurrency: int`** - Maximum concurrent requests

**Basic Example:**

```python
from chutes.templates import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium"
)
```

**Advanced Example:**

```python
from chutes.templates import build_vllm_chute
from chutes.chute import NodeSelector

# High-performance configuration
chute = build_vllm_chute(
    username="myuser",
    name="llama2-70b-chat",
    model_name="meta-llama/Llama-2-70b-chat-hf",
    image="chutes/vllm:0.9.2.dev0",
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=48,
        exclude=["l40", "a6000", "b200", "mi300x"]
    ),
    engine_args={
        "gpu_memory_utilization": 0.97,
        "max_model_len": 96000,
        "max_num_seqs": 8,
        "trust_remote_code": True,
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "tool_call_parser": "mistral",
        "enable_auto_tool_choice": True
    },
    concurrency=8
)
```

**Common VLLM Engine Arguments:**

```python
engine_args = {
    # Memory management
    "gpu_memory_utilization": 0.95,  # Use 95% of GPU memory
    "swap_space": 4,                 # GB of CPU swap space

    # Model configuration
    "max_model_len": 4096,           # Maximum sequence length
    "max_num_seqs": 256,             # Maximum concurrent sequences
    "max_paddings": 256,             # Maximum padding tokens

    # Performance optimization
    "enable_prefix_caching": True,   # Cache prefixes for efficiency
    "use_v2_block_manager": True,    # Use improved block manager
    "block_size": 16,                # Token block size

    # Model loading
    "load_format": "auto",           # Model loading format
    "revision": "main",              # Model revision/branch
    "trust_remote_code": False,      # Allow custom model code

    # Quantization
    "quantization": None,            # e.g., "awq", "gptq", "fp8"
    "dtype": "auto",                 # Model data type

    # Distributed inference
    "tensor_parallel_size": 1,       # Number of GPUs for tensor parallelism
    "pipeline_parallel_size": 1,     # Number of GPUs for pipeline parallelism

    # Tokenizer configuration
    "tokenizer_mode": "auto",        # Tokenizer mode
    "skip_tokenizer_init": False,    # Skip tokenizer initialization

    # Sampling parameters
    "seed": None,                    # Random seed
    "max_logprobs": 0,              # Maximum log probabilities to return
}
```

**Model-Specific Configurations:**

```python
# Llama 2 models
llama2_args = {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "trust_remote_code": False
}

# Mistral models
mistral_args = {
    "max_model_len": 32768,
    "tokenizer_mode": "mistral",
    "config_format": "mistral",
    "load_format": "mistral",
    "tool_call_parser": "mistral"
}

# Code Llama models
codellama_args = {
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.95,
    "enable_prefix_caching": True
}

# Yi models
yi_args = {
    "max_model_len": 4096,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.9
}
```

### SGLang Templates

#### `build_sglang_chute()`

Create a chute optimized for SGLang (structured generation language serving).

**Signature:**

```python
def build_sglang_chute(
    username: str,
    name: Optional[str] = None,
    readme: Optional[str] = None,
    model_name: str,
    image: Optional[str] = None,
    node_selector: Optional[NodeSelector] = None,
    engine_args: Optional[Dict[str, Any]] = None,
    concurrency: int = 1,
    **kwargs
) -> Chute
```

**Basic Example:**

```python
from chutes.templates import build_sglang_chute

chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium"
)
```

**Advanced Example:**

```python
from chutes.templates import build_sglang_chute
from chutes.chute import NodeSelector

chute = build_sglang_chute(
    username="myuser",
    name="deepseek-r1",
    model_name="deepseek-ai/DeepSeek-R1",
    node_selector=NodeSelector(
        gpu_count=8,
        include=["h200"],
        min_vram_gb_per_gpu=141
    ),
    engine_args={
        "host": "0.0.0.0",
        "port": 30000,
        "tp_size": 8,
        "trust_remote_code": True,
        "context_length": 65536,
        "mem_fraction_static": 0.8
    },
    concurrency=4
)
```

**Common SGLang Engine Arguments:**

```python
engine_args = {
    # Server configuration
    "host": "0.0.0.0",
    "port": 30000,

    # Model configuration
    "context_length": 4096,         # Maximum context length
    "trust_remote_code": True,      # Allow custom model code

    # Performance optimization
    "tp_size": 1,                   # Tensor parallelism size
    "mem_fraction_static": 0.9,     # Static memory fraction
    "chunked_prefill_size": 512,    # Chunked prefill size

    # Sampling configuration
    "disable_regex_jump_forward": False,  # Regex optimization
    "enable_flashinfer": True,      # FlashAttention support

    # Quantization
    "quantization": None,           # Quantization method
    "load_format": "auto",          # Loading format
}
```

### Text Embeddings Inference (TEI) Templates

#### `build_tei_chute()`

Create a chute optimized for Text Embeddings Inference.

**Signature:**

```python
def build_tei_chute(
    username: str,
    name: Optional[str] = None,
    readme: Optional[str] = None,
    model_name: str,
    image: Optional[str] = None,
    node_selector: Optional[NodeSelector] = None,
    engine_args: Optional[Dict[str, Any]] = None,
    concurrency: int = 1,
    **kwargs
) -> Chute
```

**Basic Example:**

```python
from chutes.templates import build_tei_chute

chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Advanced Example:**

```python
from chutes.templates import build_tei_chute
from chutes.chute import NodeSelector

chute = build_tei_chute(
    username="myuser",
    name="embeddings-service",
    model_name="BAAI/bge-large-en-v1.5",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
        include=["a100", "l40", "a6000"]
    ),
    engine_args={
        "max_concurrent_requests": 512,
        "max_batch_tokens": 16384,
        "max_batch_requests": 32,
        "pooling": "mean",
        "normalize": True
    },
    concurrency=16
)
```

**Common TEI Engine Arguments:**

```python
engine_args = {
    # Batching configuration
    "max_concurrent_requests": 512,    # Maximum concurrent requests
    "max_batch_tokens": 16384,         # Maximum tokens per batch
    "max_batch_requests": 32,          # Maximum requests per batch

    # Model configuration
    "pooling": "mean",                 # Pooling strategy (mean, cls, last)
    "normalize": True,                 # Normalize embeddings
    "truncate": True,                  # Truncate long sequences

    # Performance optimization
    "auto_truncate": True,             # Automatically truncate
    "default_prompt_name": None,       # Default prompt template
}
```

### Diffusion Templates

#### `build_diffusion_chute()`

Create a chute optimized for diffusion model inference (image generation).

**Signature:**

```python
def build_diffusion_chute(
    username: str,
    name: Optional[str] = None,
    readme: Optional[str] = None,
    model_name: str,
    image: Optional[str] = None,
    node_selector: Optional[NodeSelector] = None,
    engine_args: Optional[Dict[str, Any]] = None,
    concurrency: int = 1,
    **kwargs
) -> Chute
```

**Basic Example:**

```python
from chutes.templates import build_diffusion_chute

chute = build_diffusion_chute(
    username="myuser",
    model_name="black-forest-labs/FLUX.1-dev"
)
```

**Advanced Example:**

```python
from chutes.templates import build_diffusion_chute
from chutes.chute import NodeSelector

chute = build_diffusion_chute(
    username="myuser",
    name="flux-pro-service",
    model_name="black-forest-labs/FLUX.1-pro",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=80,
        include=["h100", "a100"]
    ),
    engine_args={
        "torch_dtype": "bfloat16",
        "enable_model_cpu_offload": False,
        "enable_vae_slicing": False,
        "enable_vae_tiling": False,
        "max_sequence_length": 512,
        "guidance_scale": 3.5,
        "num_inference_steps": 28
    },
    concurrency=1
)
```

**Common Diffusion Engine Arguments:**

```python
engine_args = {
    # Model optimization
    "torch_dtype": "float16",           # Model precision
    "enable_model_cpu_offload": True,   # Offload to CPU when idle
    "enable_sequential_cpu_offload": False,  # Sequential CPU offload

    # Memory optimization
    "enable_vae_slicing": True,         # VAE memory optimization
    "enable_vae_tiling": True,          # VAE tiling for large images
    "enable_attention_slicing": True,   # Attention memory optimization

    # Generation parameters
    "guidance_scale": 7.5,              # Default guidance scale
    "num_inference_steps": 50,          # Default inference steps
    "max_sequence_length": 77,          # Maximum prompt length

    # Scheduler configuration
    "scheduler": "DDIM",                # Scheduler type
    "beta_start": 0.00085,              # Scheduler beta start
    "beta_end": 0.012,                  # Scheduler beta end

    # Safety and filters
    "requires_safety_checker": True,    # Enable safety checker
    "safety_checker": None,             # Custom safety checker
    "feature_extractor": None,          # Custom feature extractor
}
```

## Template Customization

### Extending Templates

You can extend templates with additional functionality:

```python
from chutes.templates import build_vllm_chute
from chutes.chute import NodeSelector

# Start with template
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)

# Add custom endpoints
@chute.cord(public_api_path="/custom_generate", method="POST")
async def custom_generate(self, prompt: str, style: str = "formal"):
    """Custom generation with style control."""

    style_prompts = {
        "formal": "Please respond in a formal, professional tone: ",
        "casual": "Please respond in a casual, friendly tone: ",
        "technical": "Please provide a detailed technical explanation: "
    }

    styled_prompt = style_prompts.get(style, "") + prompt

    # Use the template's built-in generation
    result = await self.generate_text(styled_prompt)

    return {
        "generated_text": result["text"],
        "style": style,
        "original_prompt": prompt
    }

# Add custom initialization
@chute.on_startup()
async def custom_initialization(self):
    """Custom initialization logic."""

    # Initialize custom components
    self.style_classifier = await self.load_style_classifier()
    self.content_filter = await self.load_content_filter()

    self.logger.info("Custom components initialized")

# Add custom background job
@chute.job(name="model_health_check", schedule="*/5 * * * *")
async def health_check_job(self):
    """Monitor model health every 5 minutes."""

    try:
        # Test generation
        test_result = await self.generate_text("Hello, world!")

        # Log health status
        self.logger.info("Model health check passed")

        return {"status": "healthy", "test_passed": True}

    except Exception as e:
        self.logger.error(f"Model health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
```

### Multi-Model Templates

Combine multiple templates for complex applications:

```python
from chutes.templates import build_vllm_chute, build_tei_chute, build_diffusion_chute
from chutes.chute import Chute, NodeSelector

# Create a multi-modal AI service
class MultiModalChute:
    """Multi-modal AI service combining text, embeddings, and image generation."""

    def __init__(self, username: str):
        self.username = username

        # Text generation service
        self.text_chute = build_vllm_chute(
            username=username,
            name="text-service",
            model_name="microsoft/DialoGPT-medium",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
        )

        # Embeddings service
        self.embeddings_chute = build_tei_chute(
            username=username,
            name="embeddings-service",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=8)
        )

        # Image generation service
        self.image_chute = build_diffusion_chute(
            username=username,
            name="image-service",
            model_name="runwayml/stable-diffusion-v1-5",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24)
        )

    def get_chutes(self) -> List[Chute]:
        """Get all chutes for deployment."""
        return [self.text_chute, self.embeddings_chute, self.image_chute]

# Usage
multi_modal = MultiModalChute("myuser")
chutes = multi_modal.get_chutes()

# Deploy all services
for chute in chutes:
    # Deploy each chute
    pass
```

### Template Inheritance

Create custom templates based on existing ones:

```python
from chutes.templates import build_vllm_chute
from chutes.chute import NodeSelector
from typing import Optional, Dict, Any

def build_custom_llm_chute(
    username: str,
    model_name: str,
    name: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Chute:
    """Custom LLM template with opinionated defaults."""

    # Default configuration
    default_config = {
        "node_selector": NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=24,
            include=["a100", "l40"],
            exclude=["t4", "v100"]
        ),
        "engine_args": {
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "max_num_seqs": 128,
            "enable_prefix_caching": True
        },
        "concurrency": 4
    }

    # Merge with custom configuration
    if custom_config:
        default_config.update(custom_config)

    # Override with kwargs
    default_config.update(kwargs)

    # Create base chute with VLLM template
    chute = build_vllm_chute(
        username=username,
        name=name,
        model_name=model_name,
        **default_config
    )

    # Add custom functionality
    @chute.cord(public_api_path="/health_detailed", method="GET")
    async def detailed_health_check(self):
        """Detailed health check endpoint."""

        health_data = {
            "model_loaded": hasattr(self, 'model'),
            "gpu_available": torch.cuda.is_available(),
            "memory_usage": {},
            "timestamp": datetime.now().isoformat()
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                health_data["memory_usage"][f"gpu_{i}"] = {
                    "allocated_gb": memory_allocated / (1024**3),
                    "total_gb": memory_total / (1024**3),
                    "utilization_percent": (memory_allocated / memory_total) * 100
                }

        return health_data

    return chute

# Usage
custom_chute = build_custom_llm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    custom_config={
        "concurrency": 8,
        "engine_args": {
            "max_model_len": 8192  # Override default
        }
    }
)
```

## Template Configuration Patterns

### Development vs Production

```python
def create_development_chute(username: str, model_name: str) -> Chute:
    """Development-optimized configuration."""

    return build_vllm_chute(
        username=username,
        name=f"dev-{model_name.split('/')[-1]}",
        model_name=model_name,
        node_selector=NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=8,
            include=["t4", "v100", "l40"]),
        engine_args={
            "gpu_memory_utilization": 0.8,  # Conservative for stability
            "max_model_len": 2048,           # Smaller context for speed
            "max_num_seqs": 32               # Lower concurrency
        },
        concurrency=2
    )

def create_production_chute(username: str, model_name: str) -> Chute:
    """Production-optimized configuration."""

    return build_vllm_chute(
        username=username,
        name=f"prod-{model_name.split('/')[-1]}",
        model_name=model_name,
        node_selector=NodeSelector(
            gpu_count=2,
            min_vram_gb_per_gpu=40,
            include=["a100", "h100"],
            exclude=["t4", "v100"]),
        engine_args={
            "gpu_memory_utilization": 0.95,  # Maximum utilization
            "max_model_len": 4096,            # Full context length
            "max_num_seqs": 256,              # High concurrency
            "enable_prefix_caching": True,    # Performance optimization
            "use_v2_block_manager": True
        },
        concurrency=8
    )
```

### Model Size-Based Configuration

```python
def create_chute_for_model_size(
    username: str,
    model_name: str,
    model_size: str
) -> Chute:
    """Create chute optimized for specific model sizes."""

    configurations = {
        "small": {  # < 1B parameters
            "node_selector": NodeSelector(
                gpu_count=1,
                min_vram_gb_per_gpu=8,
                include=["t4", "v100", "l40"]
            ),
            "engine_args": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 2048,
                "max_num_seqs": 128
            },
            "concurrency": 8
        },

        "medium": {  # 1B-10B parameters
            "node_selector": NodeSelector(
                gpu_count=1,
                min_vram_gb_per_gpu=24,
                include=["a100", "l40", "a6000"]
            ),
            "engine_args": {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "max_num_seqs": 256
            },
            "concurrency": 4
        },

        "large": {  # 10B-50B parameters
            "node_selector": NodeSelector(
                gpu_count=2,
                min_vram_gb_per_gpu=40,
                include=["a100", "h100"]
            ),
            "engine_args": {
                "gpu_memory_utilization": 0.95,
                "max_model_len": 4096,
                "max_num_seqs": 128,
                "tensor_parallel_size": 2
            },
            "concurrency": 2
        },

        "xlarge": {  # 50B+ parameters
            "node_selector": NodeSelector(
                gpu_count=8,
                min_vram_gb_per_gpu=80,
                include=["h100", "a100"]
            ),
            "engine_args": {
                "gpu_memory_utilization": 0.95,
                "max_model_len": 4096,
                "max_num_seqs": 64,
                "tensor_parallel_size": 8
            },
            "concurrency": 1
        }
    }

    config = configurations.get(model_size, configurations["medium"])

    return build_vllm_chute(
        username=username,
        name=f"{model_size}-{model_name.split('/')[-1]}",
        model_name=model_name,
        **config
    )
```

## Template Testing and Validation

### Template Testing Framework

```python
import pytest
from chutes.templates import build_vllm_chute, build_tei_chute
from chutes.chute import NodeSelector

class TemplateTestSuite:
    """Test suite for template validation."""

    def test_vllm_template_basic(self):
        """Test basic VLLM template creation."""

        chute = build_vllm_chute(
            username="test",
            model_name="microsoft/DialoGPT-medium"
        )

        assert chute.config.username == "test"
        assert chute.config.name is not None
        assert hasattr(chute, 'generate_text')  # Should have generation endpoint

    def test_vllm_template_with_custom_config(self):
        """Test VLLM template with custom configuration."""

        node_selector = NodeSelector(
            gpu_count=2,
            min_vram_gb_per_gpu=24
        )

        engine_args = {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9
        }

        chute = build_vllm_chute(
            username="test",
            name="custom-llm",
            model_name="microsoft/DialoGPT-medium",
            node_selector=node_selector,
            engine_args=engine_args,
            concurrency=4
        )

        assert chute.config.name == "custom-llm"
        assert chute.config.concurrency == 4
        assert chute.config.node_selector.gpu_count == 2

    def test_tei_template_basic(self):
        """Test basic TEI template creation."""

        chute = build_tei_chute(
            username="test",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        assert chute.config.username == "test"
        assert hasattr(chute, 'embed_text')  # Should have embedding endpoint

    def test_template_validation(self):
        """Test template parameter validation."""

        # Test invalid parameters
        with pytest.raises(ValueError):
            build_vllm_chute(
                username="",  # Invalid username
                model_name="microsoft/DialoGPT-medium"
            )

        with pytest.raises(ValueError):
            build_vllm_chute(
                username="test",
                model_name=""  # Invalid model name
            )

    def test_template_consistency(self):
        """Test that templates produce consistent configurations."""

        chute1 = build_vllm_chute(
            username="test",
            model_name="microsoft/DialoGPT-medium"
        )

        chute2 = build_vllm_chute(
            username="test",
            model_name="microsoft/DialoGPT-medium"
        )

        # Should have identical configurations
        assert chute1.config.username == chute2.config.username
        assert chute1.config.concurrency == chute2.config.concurrency

# Run template tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Best Practices

### Template Selection Guidelines

1. **Choose the Right Template**

   ```python
   # For language models with OpenAI-compatible API
   vllm_chute = build_vllm_chute(username="user", model_name="gpt2")

   # For structured generation and complex prompting
   sglang_chute = build_sglang_chute(username="user", model_name="gpt2")

   # For text embeddings and similarity search
   tei_chute = build_tei_chute(username="user", model_name="sentence-transformers/all-MiniLM-L6-v2")

   # For image generation
   diffusion_chute = build_diffusion_chute(username="user", model_name="runwayml/stable-diffusion-v1-5")
   ```

2. **Start with Templates, Customize as Needed**

   ```python
   # Start with template
   chute = build_vllm_chute(username="user", model_name="gpt2")

   # Add custom functionality
   @chute.cord(public_api_path="/custom")
   async def custom_endpoint(self):
       return {"message": "Custom functionality"}
   ```

3. **Use Appropriate Node Selectors**

   ```python
   # Match hardware to model requirements
   large_model_chute = build_vllm_chute(
       username="user",
       model_name="meta-llama/Llama-2-70b-hf",
       node_selector=NodeSelector(
           gpu_count=4,
           min_vram_gb_per_gpu=48
       )
   )
   ```

4. **Optimize Engine Arguments**
   ```python
   # Production optimization
   production_chute = build_vllm_chute(
       username="user",
       model_name="gpt2",
       engine_args={
           "gpu_memory_utilization": 0.95,
           "enable_prefix_caching": True,
           "use_v2_block_manager": True
       }
   )
   ```

### Common Patterns

1. **Environment-Specific Configuration**

   ```python
   def create_chute_for_environment(env: str, username: str, model_name: str):
       if env == "development":
           return build_vllm_chute(
               username=username,
               model_name=model_name,
               node_selector=NodeSelector(),
               concurrency=1
           )
       elif env == "production":
           return build_vllm_chute(
               username=username,
               model_name=model_name,
               node_selector=NodeSelector(),
               concurrency=8
           )
   ```

2. **A/B Testing Templates**

   ```python
   # Template A: VLLM
   template_a = build_vllm_chute(
       username="user",
       name="model-a",
       model_name="microsoft/DialoGPT-medium"
   )

   # Template B: SGLang
   template_b = build_sglang_chute(
       username="user",
       name="model-b",
       model_name="microsoft/DialoGPT-medium"
   )
   ```

3. **Template Composition**
   ```python
   def create_complete_ai_service(username: str):
       return {
           "text": build_vllm_chute(username, model_name="gpt2"),
           "embeddings": build_tei_chute(username, model_name="sentence-transformers/all-MiniLM-L6-v2"),
           "images": build_diffusion_chute(username, model_name="runwayml/stable-diffusion-v1-5")
       }
   ```

This comprehensive guide covers all aspects of Chutes templates for rapid AI application deployment.
