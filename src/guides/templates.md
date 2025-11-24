# Using Pre-built Templates

This guide covers how to effectively use Chutes' pre-built templates to rapidly deploy AI applications with minimal configuration while maintaining flexibility for customization.

## Overview

Pre-built templates provide:

- **Rapid Deployment**: Get AI models running in minutes
- **Best Practices**: Optimized configurations and performance tuning
- **Proven Architectures**: Battle-tested model serving patterns
- **Easy Customization**: Modify templates to fit your needs
- **Production Ready**: Built-in scaling, monitoring, and error handling

## Available Templates

### VLLM Template

High-performance large language model serving with OpenAI compatibility.

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

# Basic VLLM deployment
chute = build_vllm_chute(
    username="myuser",
    readme="microsoft/DialoGPT-medium for conversational AI",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    ),
    concurrency=4
)
```

**Key Features:**

- OpenAI-compatible API endpoints
- Automatic batching and CUDA graph optimization
- Support for all major open-source LLMs
- Built-in streaming and function calling
- Multi-GPU distributed inference

### SGLang Template

Advanced structured generation with programmable text generation.

```python
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="myuser",
    readme="Qwen2.5-7B-Instruct with SGLang",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    concurrency=8
)
```

**Key Features:**

- Advanced structured generation
- Custom sampling and constraints
- Batch processing optimizations
- Memory-efficient serving
- Real-time streaming responses

### TEI Template (Text Embeddings Inference)

High-performance text embedding generation for similarity search and RAG.

```python
from chutes.chute import NodeSelector
from chutes.chute.template.tei import build_tei_chute

chute = build_tei_chute(
    username="myuser",
    readme="sentence-transformers/all-MiniLM-L6-v2 embeddings",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8
    ),
    concurrency=16
)
```

**Key Features:**

- Optimized embedding generation
- Batch processing for efficiency
- Multiple pooling strategies
- Built-in similarity computation
- Support for various embedding models

### Diffusion Template

Image generation using state-of-the-art diffusion models.

```python
from chutes.chute import NodeSelector
from chutes.chute.template.diffusion import build_diffusion_chute

chute = build_diffusion_chute(
    username="myuser",
    readme="Stable Diffusion XL for image generation",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    ),
    concurrency=2
)
```

**Key Features:**

- Support for various diffusion architectures
- Text-to-image and image-to-image generation
- Optimized memory usage and inference
- Built-in image processing and validation
- Support for ControlNet and LoRA

## Template Customization

### Basic Parameter Tuning

All templates support common parameters for customization:

```python
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

# Customized VLLM deployment
chute = build_vllm_chute(
    username="myuser",
    readme="Customized Llama 2 deployment",
    model_name="meta-llama/Llama-2-7b-chat-hf",

    # Hardware configuration
    node_selector=NodeSelector(
        gpu_count=2,                    # Multi-GPU setup
        min_vram_gb_per_gpu=40,        # High memory requirement
        include=["h100", "a100"],      # Prefer specific GPU types
        exclude=["k80", "v100"]        # Exclude older GPUs
    ),

    # Performance settings
    concurrency=8,                     # Handle 8 concurrent requests

    # Model-specific arguments
    engine_args=dict(
        gpu_memory_utilization=0.95,   # Use 95% of GPU memory
        max_model_len=4096,            # Context length
        max_num_seqs=16,               # Batch size
        temperature=0.7,               # Default temperature
        trust_remote_code=True,        # Enable custom models
        quantization="awq",            # Use AWQ quantization
        tensor_parallel_size=2,        # Use both GPUs
    ),

    # Custom image (optional)
    image="chutes/vllm:0.8.0",

    # Revision pinning for reproducibility
    revision="main"
)
```

### Advanced Engine Configuration

#### VLLM Advanced Settings

```python
# Production VLLM configuration
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/WizardLM-2-8x22B",
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=80,
        include=["h100", "h200"]
    ),
    engine_args=dict(
        # Memory optimization
        gpu_memory_utilization=0.97,
        cpu_offload_gb=0,

        # Performance tuning
        max_model_len=32768,
        max_num_seqs=32,
        max_paddings=256,

        # Advanced features
        enable_prefix_caching=True,
        use_v2_block_manager=True,
        enable_chunked_prefill=True,

        # Model loading
        load_format="auto",
        dtype="auto",
        quantization="fp8",

        # Distributed settings
        tensor_parallel_size=8,
        pipeline_parallel_size=1,

        # API compatibility
        served_model_name="wizardlm-2-8x22b",
        chat_template="chatml",

        # Logging and monitoring
        disable_log_requests=False,
        max_log_len=2048),
    concurrency=16
)
```

#### SGLang Optimization

```python
# Optimized SGLang configuration
chute = build_sglang_chute(
    username="myuser",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    engine_args=(
        "--host 0.0.0.0 "
        "--port 30000 "
        "--model-path mistralai/Mistral-7B-Instruct-v0.2 "
        "--tokenizer-path mistralai/Mistral-7B-Instruct-v0.2 "
        "--context-length 32768 "
        "--mem-fraction-static 0.9 "
        "--tp-size 1 "
        "--stream-interval 1 "
        "--disable-flashinfer "  # For compatibility
        "--trust-remote-code"
    ),
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    )
)
```

### Custom Images with Templates

You can combine templates with custom images for additional dependencies:

```python
from chutes.image import Image
from chutes.chute.template.vllm import build_vllm_chute

# Build custom image with additional packages
custom_image = (
    Image(username="myuser", name="custom-vllm", tag="1.0")
    .from_base("chutes/vllm:0.8.0")
    .run_command("pip install langchain openai tiktoken")
    .run_command("pip install numpy pandas matplotlib")
    .with_env("CUSTOM_CONFIG", "production")
)

# Use custom image with template
chute = build_vllm_chute(
    username="myuser",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    image=custom_image,  # Use our custom image
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24
    )
)
```

## Template Patterns

### Multi-Model Deployment

Deploy multiple models using templates:

```python
# Deploy different models for different use cases
from chutes.chute.template.vllm import build_vllm_chute
from chutes.chute.template.tei import build_tei_chute

# Chat model
chat_chute = build_vllm_chute(
    username="myuser",
    name="chat-service",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)

# Code model
code_chute = build_vllm_chute(
    username="myuser",
    name="code-service",
    model_name="codellama/CodeLlama-7b-Python-hf",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)

# Embedding model
embedding_chute = build_tei_chute(
    username="myuser",
    name="embedding-service",
    model_name="sentence-transformers/all-mpnet-base-v2",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=8)
)
```

### Template Inheritance and Extension

Create your own template patterns based on existing ones:

```python
from chutes.chute.template.vllm import build_vllm_chute
from chutes.chute import NodeSelector
from chutes.image import Image

def build_chat_template(
    username: str,
    model_name: str,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
):
    """Custom template for chat applications."""

    # Custom image with chat-specific tools
    image = (
        Image(username=username, name="chat-optimized", tag="1.0")
        .from_base("chutes/vllm:latest")
        .run_command("pip install tiktoken langchain")
        .with_env("SYSTEM_PROMPT", system_prompt)
        .with_env("CHAT_MODE", "true")
    )

    # Default settings optimized for chat
    default_engine_args = {
        "max_model_len": 8192,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 1024,
        "stream": True
    }

    # Merge with user-provided args
    engine_args = kwargs.pop("engine_args", {})
    engine_args = {**default_engine_args, **engine_args}

    return build_vllm_chute(
        username=username,
        model_name=model_name,
        image=image,
        engine_args=engine_args,
        **kwargs
    )

# Use custom template
chat_chute = build_chat_template(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    system_prompt="You are a friendly customer service assistant.",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)
```

### Template-Based Microservices

Build a complete AI system using multiple templates:

```python
# microservices_deployment.py
from chutes.chute.template.vllm import build_vllm_chute
from chutes.chute.template.tei import build_tei_chute
from chutes.chute.template.diffusion import build_diffusion_chute

class AIServiceSuite:
    """Complete AI service suite using templates."""

    def __init__(self, username: str):
        self.username = username
        self.services = {}

    def deploy_text_services(self):
        """Deploy text processing services."""

        # Main chat model
        self.services["chat"] = build_vllm_chute(
            username=self.username,
            name="chat-llm",
            model_name="microsoft/DialoGPT-medium",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24),
            concurrency=8
        )

        # Specialized reasoning model
        self.services["reasoning"] = build_vllm_chute(
            username=self.username,
            name="reasoning-llm",
            model_name="deepseek-ai/deepseek-llm-7b-chat",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
            concurrency=4
        )

        # Embeddings for RAG
        self.services["embeddings"] = build_tei_chute(
            username=self.username,
            name="text-embeddings",
            model_name="sentence-transformers/all-mpnet-base-v2",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=8),
            concurrency=16
        )

    def deploy_multimodal_services(self):
        """Deploy multimodal AI services."""

        # Image generation
        self.services["image_gen"] = build_diffusion_chute(
            username=self.username,
            name="image-generator",
            model_name="stabilityai/stable-diffusion-xl-base-1.0",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24),
            concurrency=2
        )

        # Vision-language model
        self.services["vision"] = build_vllm_chute(
            username=self.username,
            name="vision-llm",
            model_name="llava-hf/llava-1.5-7b-hf",
            node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
            concurrency=4
        )

    def get_deployment_script(self):
        """Generate deployment script for all services."""

        script_lines = ["#!/bin/bash", "set -e", ""]

        for service_name, chute in self.services.items():
            script_lines.extend([
                f"echo 'Deploying {service_name}...'",
                f"chutes deploy {chute.name}:chute --wait",
                f"echo '{service_name} deployed successfully'",
                ""
            ])

        return "\n".join(script_lines)

# Usage
suite = AIServiceSuite("myuser")
suite.deploy_text_services()
suite.deploy_multimodal_services()

# Generate deployment script
deployment_script = suite.get_deployment_script()
with open("deploy_ai_suite.sh", "w") as f:
    f.write(deployment_script)
```

## Template Configuration Best Practices

### 1. Hardware Selection

Choose appropriate hardware for each template:

```python
# Memory requirements by model size
hardware_configs = {
    "small_models": {  # <7B parameters
        "node_selector": NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=16,
            include=["rtx4090", "a40", "l40"]
        ),
        "concurrency": 8
    },

    "medium_models": {  # 7B-30B parameters
        "node_selector": NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=48,
            include=["a100", "h100"]
        ),
        "concurrency": 4
    },

    "large_models": {  # 30B+ parameters
        "node_selector": NodeSelector(
            gpu_count=2,
            min_vram_gb_per_gpu=80,
            include=["h100", "h200"]
        ),
        "concurrency": 2
    }
}

def select_hardware(model_name: str):
    """Select hardware configuration based on model."""

    # Simple heuristic based on model name
    if "7b" in model_name.lower():
        return hardware_configs["small_models"]
    elif any(size in model_name.lower() for size in ["13b", "30b"]):
        return hardware_configs["medium_models"]
    else:
        return hardware_configs["large_models"]
```

### 2. Environment-Specific Configurations

```python
import os

def get_config_for_environment(env: str = "production"):
    """Get configuration based on deployment environment."""

    configs = {
        "development": {
            "concurrency": 2,
            "engine_args": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 2048,
                "disable_log_requests": False
            }
        },

        "staging": {
            "concurrency": 4,
            "engine_args": {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
                "disable_log_requests": False
            }
        },

        "production": {
            "concurrency": 8,
            "engine_args": {
                "gpu_memory_utilization": 0.95,
                "max_model_len": 8192,
                "disable_log_requests": True,
                "enable_prefix_caching": True
            }
        }
    }

    return configs.get(env, configs["production"])

# Usage
env = os.getenv("DEPLOYMENT_ENV", "production")
config = get_config_for_environment(env)

chute = build_vllm_chute(
    username="myuser",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    **config
)
```

### 3. Model-Specific Optimizations

```python
def get_model_optimizations(model_name: str):
    """Get model-specific optimizations."""

    optimizations = {
        # Llama models
        "llama": {
            "engine_args": {
                "quantization": "awq",
                "enable_prefix_caching": True,
                "use_v2_block_manager": True
            }
        },

        # Mistral models
        "mistral": {
            "engine_args": {
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "trust_remote_code": True
            }
        },

        # CodeLlama models
        "code": {
            "engine_args": {
                "max_model_len": 16384,  # Longer context for code
                "temperature": 0.1,      # Lower temperature for code
                "enable_prefix_caching": True
            }
        },

        # Chat models
        "chat": {
            "engine_args": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_tokens": 2048,
                "stream": True
            }
        }
    }

    # Detect model type from name
    model_lower = model_name.lower()

    if "llama" in model_lower:
        return optimizations["llama"]
    elif "mistral" in model_lower:
        return optimizations["mistral"]
    elif "code" in model_lower:
        return optimizations["code"]
    elif any(term in model_lower for term in ["chat", "instruct", "dialog"]):
        return optimizations["chat"]
    else:
        return {"engine_args": {}}

# Usage
model_name = "codellama/CodeLlama-7b-Python-hf"
optimizations = get_model_optimizations(model_name)

chute = build_vllm_chute(
    username="myuser",
    model_name=model_name,
    **optimizations
)
```

## Monitoring and Debugging Templates

### Template Health Checks

```python
import requests
import time

async def check_template_health(chute_url: str, template_type: str):
    """Check health of deployed template."""

    health_checks = {
        "vllm": {
            "endpoint": "/v1/models",
            "expected_status": 200
        },
        "sglang": {
            "endpoint": "/health",
            "expected_status": 200
        },
        "tei": {
            "endpoint": "/health",
            "expected_status": 200
        },
        "diffusion": {
            "endpoint": "/health",
            "expected_status": 200
        }
    }

    if template_type not in health_checks:
        return {"status": "unknown", "error": "Unknown template type"}

    check_config = health_checks[template_type]

    try:
        response = requests.get(
            f"{chute_url}{check_config['endpoint']}",
            timeout=10
        )

        if response.status_code == check_config["expected_status"]:
            return {"status": "healthy", "response_time": response.elapsed.total_seconds()}
        else:
            return {"status": "unhealthy", "status_code": response.status_code}

    except Exception as e:
        return {"status": "error", "error": str(e)}

# Usage
health = await check_template_health(
    "https://myuser-my-model.chutes.ai",
    "vllm"
)
print(f"Service health: {health}")
```

### Performance Monitoring

```python
def monitor_template_performance(chute_name: str, duration_minutes: int = 60):
    """Monitor template performance over time."""

    import subprocess
    import json

    # Collect metrics
    metrics_cmd = f"chutes chutes metrics {chute_name} --duration {duration_minutes}m --format json"
    result = subprocess.run(metrics_cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        metrics = json.loads(result.stdout)

        # Analyze metrics
        analysis = {
            "avg_response_time": metrics.get("avg_response_time", 0),
            "request_count": metrics.get("request_count", 0),
            "error_rate": metrics.get("error_rate", 0),
            "gpu_utilization": metrics.get("gpu_utilization", 0),
            "memory_usage": metrics.get("memory_usage", 0)
        }

        # Performance recommendations
        recommendations = []

        if analysis["avg_response_time"] > 5:
            recommendations.append("Consider increasing concurrency or using faster GPUs")

        if analysis["gpu_utilization"] < 50:
            recommendations.append("GPU underutilized - consider reducing instance size")

        if analysis["error_rate"] > 5:
            recommendations.append("High error rate - check logs and model configuration")

        return {
            "metrics": analysis,
            "recommendations": recommendations
        }

    else:
        return {"error": "Failed to collect metrics", "details": result.stderr}
```

## Template Migration and Updates

### Upgrading Template Versions

```python
def upgrade_template_safely(
    current_chute_name: str,
    new_template_version: str,
    model_name: str,
    username: str
):
    """Safely upgrade a template to a new version."""

    # Create new chute with updated template
    staging_name = f"{current_chute_name}-staging"

    new_chute = build_vllm_chute(
        username=username,
        name=staging_name,
        model_name=model_name,
        image=f"chutes/vllm:{new_template_version}",
        # Copy current configuration
        node_selector=get_current_node_selector(current_chute_name),
        engine_args=get_current_engine_args(current_chute_name)
    )

    # Deployment script
    upgrade_script = f"""
    # Deploy staging version
    chutes deploy {staging_name}:chute --wait

    # Test staging deployment
    python test_template.py --target {staging_name}

    # If tests pass, switch traffic
    if [ $? -eq 0 ]; then
        echo "Tests passed, deploying to production"
        chutes deploy {current_chute_name}:chute --wait
        chutes chutes delete {staging_name}
    else
        echo "Tests failed, keeping current version"
        chutes chutes delete {staging_name}
    fi
    """

    return upgrade_script
```

## Troubleshooting Templates

### Common Issues and Solutions

```python
def diagnose_template_issues(chute_name: str, template_type: str):
    """Diagnose common template deployment issues."""

    issues = []

    # Check deployment status
    status_cmd = f"chutes chutes get {chute_name}"
    status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)

    if "Failed" in status_result.stdout:
        issues.append({
            "issue": "Deployment failed",
            "solution": "Check logs with: chutes chutes logs " + chute_name
        })

    # Check resource usage
    metrics_cmd = f"chutes chutes metrics {chute_name}"
    metrics_result = subprocess.run(metrics_cmd, shell=True, capture_output=True, text=True)

    if "OutOfMemory" in metrics_result.stdout:
        issues.append({
            "issue": "GPU out of memory",
            "solution": "Reduce gpu_memory_utilization or increase GPU size"
        })

    # Template-specific checks
    if template_type == "vllm":
        # Check for VLLM-specific issues
        if "CUDA_ERROR_OUT_OF_MEMORY" in metrics_result.stdout:
            issues.append({
                "issue": "VLLM CUDA memory error",
                "solution": "Reduce max_model_len or batch size (max_num_seqs)"
            })

    elif template_type == "sglang":
        # Check for SGLang-specific issues
        if "RuntimeError" in metrics_result.stdout:
            issues.append({
                "issue": "SGLang runtime error",
                "solution": "Check model compatibility and reduce memory usage"
            })

    return issues

# Quick diagnostics
issues = diagnose_template_issues("my-llm-service", "vllm")
for issue in issues:
    print(f"Issue: {issue['issue']}")
    print(f"Solution: {issue['solution']}\n")
```

## Next Steps

- **Custom Templates**: Build your own reusable templates
- **Production Scaling**: Monitor and optimize template performance
- **Advanced Patterns**: Combine templates for complex architectures
- **CI/CD Integration**: Automate template deployments

For more advanced topics, see:

- [Custom Chutes Guide](custom-chutes)
- [Performance Optimization](performance-optimization)
- [Production Best Practices](best-practices)
