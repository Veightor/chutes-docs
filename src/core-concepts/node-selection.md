# Node Selection (Hardware)

**Node Selection** in Chutes allows you to specify exactly what hardware your application needs. This ensures optimal performance while controlling costs by only using the GPU resources you actually need.

## What is Node Selection?

Node Selection defines the hardware requirements for your chute:

- 🖥️ **GPU type and count** (A100, H100, V100, etc.)
- 💾 **VRAM requirements** per GPU
- 🔧 **CPU and memory** specifications
- 🎯 **Hardware preferences** (include/exclude specific types)
- 🌍 **Geographic regions** for deployment

## Basic Node Selection

```python
from chutes.chute import NodeSelector, Chute

# Simple GPU requirement
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16
)

chute = Chute(
    username="myuser",
    name="my-chute",
    image="my-image",
    node_selector=node_selector
)
```

## NodeSelector Parameters

### GPU Requirements

#### `gpu_count: int`

Number of GPUs your application needs.

```python
# Single GPU for small models
NodeSelector(gpu_count=1)

# Multi-GPU for large models
NodeSelector(gpu_count=4)

# Maximum parallelization
NodeSelector(gpu_count=8)
```

#### `min_vram_gb_per_gpu: int`

Minimum VRAM (video memory) required per GPU.

```python
# Small models (e.g., BERT, small LLMs)
NodeSelector(min_vram_gb_per_gpu=8)

# Medium models (e.g., 7B parameter models)
NodeSelector(min_vram_gb_per_gpu=16)

# Large models (e.g., 13B+ parameter models)
NodeSelector(min_vram_gb_per_gpu=24)

# Very large models (e.g., 70B+ parameter models)
NodeSelector(min_vram_gb_per_gpu=80)
```

### Hardware Preferences

#### `include: list[str] = None`

Prefer specific GPU types or categories.

```python
# Prefer latest generation GPUs
NodeSelector(include=["a100", "h100"])

# Prefer high-memory GPUs
NodeSelector(include=["a100_80gb", "h100_80gb"])

# Include budget-friendly options
NodeSelector(include=["rtx4090", "rtx3090"])
```

#### `exclude: list[str] = None`

Avoid specific GPU types or categories.

```python
# Avoid older generation GPUs
NodeSelector(exclude=["k80", "p100", "v100"])

# Avoid specific models
NodeSelector(exclude=["rtx3080", "rtx2080"])

# Avoid low-memory variants
NodeSelector(exclude=["a100_40gb"])
```

### CPU and Memory

#### `min_cpu_count: int = None`

Minimum CPU cores required.

```python
# CPU-intensive preprocessing
NodeSelector(min_cpu_count=16)

# Heavy data loading
NodeSelector(min_cpu_count=32)
```

#### `min_memory_gb: int = None`

Minimum system RAM required.

```python
# Large dataset in memory
NodeSelector(min_memory_gb=64)

# Very large preprocessing
NodeSelector(min_memory_gb=256)
```

### Geographic Preferences

#### `regions: list[str] = None`

Preferred deployment regions.

```python
# US regions only
NodeSelector(regions=["us-east", "us-west"])

# Europe regions
NodeSelector(regions=["eu-west", "eu-central"])

# Global deployment
NodeSelector(regions=["us-east", "eu-west", "asia-pacific"])
```

## Common Hardware Configurations

### Small Language Models (< 1B parameters)

```python
# BERT, DistilBERT, small T5 models
small_model_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8
)
```

### Medium Language Models (1B - 7B parameters)

```python
# GPT-2, small LLaMA models, Flan-T5
medium_model_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["rtx4090", "a100", "h100"]
)
```

### Large Language Models (7B - 30B parameters)

```python
# LLaMA 7B-13B, GPT-3 variants
large_model_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["a100", "h100"],
    exclude=["rtx3080", "rtx4080"]  # Not enough VRAM
)
```

### Very Large Language Models (30B+ parameters)

```python
# LLaMA 30B+, GPT-4 class models
xl_model_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["a100_80gb", "h100_80gb"]
)
```

### Massive Models (100B+ parameters)

```python
# Very large models requiring model parallelism
massive_model_selector = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80,
    include=["a100_80gb", "h100_80gb"],
    min_cpu_count=64,
    min_memory_gb=512
)
```

## GPU Types and Specifications

### NVIDIA A100

```python
# A100 40GB - excellent for most workloads
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=40,
    include=["a100_40gb"]
)

# A100 80GB - for very large models
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=80,
    include=["a100_80gb"]
)
```

### NVIDIA H100

```python
# Latest generation, highest performance
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=80,
    include=["h100"]
)
```

### RTX Series (Cost-Effective)

```python
# RTX 4090 - excellent price/performance
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["rtx4090"]
)

# RTX 3090 - budget option
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["rtx3090"]
)
```

### V100 (Legacy but Stable)

```python
# V100 for proven workloads
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["v100"]
)
```

## Advanced Selection Strategies

### Cost Optimization

```python
# Prefer cost-effective GPUs
cost_optimized = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["rtx4090", "rtx3090", "v100"],
    exclude=["a100", "h100"]  # More expensive
)
```

### Performance Optimization

```python
# Prefer highest performance
performance_optimized = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100_80gb"],
    exclude=["rtx", "v100"]  # Lower performance
)
```

### Availability Optimization

```python
# Prefer widely available hardware
availability_optimized = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["rtx4090", "a100", "v100"],
    regions=["us-east", "us-west", "eu-west"]
)
```

### Multi-Region Deployment

```python
# Global deployment with failover
global_deployment = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["a100", "h100"],
    regions=["us-east", "us-west", "eu-west", "asia-pacific"]
)
```

## Memory Requirements by Use Case

### Text Generation

```python
# Small models (up to 7B parameters)
text_gen_small = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16
)

# Large models (7B-30B parameters)
text_gen_large = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40
)
```

### Image Generation

```python
# Stable Diffusion variants
image_gen = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=12,  # SD 1.5/2.1
    include=["rtx4090", "a100"]
)

# High-resolution image generation
image_gen_hires = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,  # SDXL, custom models
    include=["rtx4090", "a100"]
)
```

### Video Processing

```python
# Video analysis and generation
video_processing = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=24,
    min_cpu_count=16,
    min_memory_gb=64
)
```

### Training Workloads

```python
# Model fine-tuning
training_workload = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=40,
    min_cpu_count=32,
    min_memory_gb=128,
    include=["a100", "h100"]
)
```

## Template-Specific Recommendations

### VLLM Template

```python
from chutes.chute.template.vllm import build_vllm_chute

# Optimized for VLLM inference
vllm_chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
        include=["a100", "h100", "rtx4090"]  # VLLM optimized
    )
)
```

### Diffusion Template

```python
from chutes.chute.template.diffusion import build_diffusion_chute

# Optimized for image generation
diffusion_chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=12,
        include=["rtx4090", "a100"]  # Good for image gen
    )
)
```

## Best Practices

### 1. Start Conservative

```python
# Begin with minimum requirements
conservative_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16
)

# Scale up if needed
```

### 2. Test Different Configurations

```python
# Development configuration
dev_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8,
    include=["rtx3090", "rtx4090"]
)

# Production configuration
prod_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40,
    include=["a100", "h100"]
)
```

### 3. Consider Cost vs Performance

```python
# Budget-conscious
budget_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["rtx4090", "v100"],
    exclude=["a100", "h100"]
)

# Performance-critical
performance_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100_80gb"]
)
```

### 4. Plan for Scaling

```python
# Single instance
single_instance = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)

# Multi-instance ready
multi_instance = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    regions=["us-east", "us-west", "eu-west"]
)
```

## Monitoring and Optimization

### Resource Utilization

Monitor your chute's actual resource usage:

```python
# Over-provisioned (waste of money)
over_provisioned = NodeSelector(
    gpu_count=4,  # Using only 1
    min_vram_gb_per_gpu=80  # Using only 20GB
)

# Right-sized (cost-effective)
right_sized = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)
```

### Performance Tuning

```python
# CPU-bound preprocessing
cpu_intensive = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    min_cpu_count=16,  # Extra CPU for preprocessing
    min_memory_gb=64
)

# GPU-bound inference
gpu_intensive = NodeSelector(
    gpu_count=2,  # More GPU power
    min_vram_gb_per_gpu=40,
    min_cpu_count=8   # Less CPU needed
)
```

## Troubleshooting

### Common Issues

#### "No available nodes"

```python
# Too restrictive
problematic = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80,
    include=["h100"],
    regions=["specific-rare-region"]
)

# More flexible
flexible = NodeSelector(
    gpu_count=4,  # Reduced requirement
    min_vram_gb_per_gpu=40,
    include=["h100", "a100_80gb"],  # More options
    regions=["us-east", "us-west"]  # More regions
)
```

#### "High costs"

```python
# Expensive configuration
expensive = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80,
    include=["h100"]
)

# Cost-optimized alternative
cost_optimized = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40,
    include=["a100", "rtx4090"]
)
```

#### "Poor performance"

```python
# Underpowered
underpowered = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8,
    include=["rtx3080"]
)

# Better performance
better_performance = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["rtx4090", "a100"]
)
```

## Next Steps

- **[Chutes](/docs/core-concepts/chutes)** - Learn how to use NodeSelector in Chutes
- **[Templates](/docs/core-concepts/templates)** - Pre-configured hardware for common use cases
- **[Best Practices Guide](/docs/guides/best-practices)** - Optimization strategies
- **[Cost Management](/docs/guides/cost-optimization)** - Control and optimize costs
