# NodeSelector API Reference

The `NodeSelector` class specifies hardware requirements for Chutes deployments. This reference covers all configuration options, GPU types, and best practices for optimal resource allocation.

## Class Definition

```python
from chutes.chute import NodeSelector

node_selector = NodeSelector(
    gpu_count: int = 1,
    min_vram_gb_per_gpu: int = 16,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
)
```

## Parameters

### GPU Configuration

#### `gpu_count: int = 1`

Number of GPUs required for the deployment. Valid range: 1-8 GPUs.

**Examples:**

```python
# Single GPU (default)
node_selector = NodeSelector(gpu_count=1)

# Multiple GPUs for large models
node_selector = NodeSelector(gpu_count=4)

# Maximum supported GPUs
node_selector = NodeSelector(gpu_count=8)
```

**Use Cases:**

- **1 GPU**: Most standard AI models (BERT, GPT-2, small LLMs)
- **2-4 GPUs**: Larger language models (7B-30B parameters)
- **4-8 GPUs**: Very large models (70B+ parameters, distributed inference)

#### `min_vram_gb_per_gpu: int = 16`

Minimum VRAM (Video RAM) required per GPU in gigabytes. Valid range: 16-140 GB.

**Examples:**

```python
# Default minimum (suitable for most models)
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16
)

# Medium models requiring more VRAM
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)

# Large models
node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=48
)

# Ultra-large models
node_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=80
)
```

**VRAM Requirements by Model Size:**

- **1-3B parameters**: 16GB VRAM
- **7B parameters**: 24GB VRAM
- **13B parameters**: 32-40GB VRAM
- **30B parameters**: 48GB VRAM
- **70B+ parameters**: 80GB+ VRAM (often requires multiple GPUs)

### GPU Type Filtering

#### `include: Optional[List[str]] = None`

List of GPU types to include in selection. If specified, only these GPU types will be considered.

**Examples:**

```python
# Prefer specific GPU types
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["a100", "h100"]  # Only A100 or H100 GPUs
)

# Target cost-effective options
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48,
    include=["l40", "a6000"]  # L40 or A6000 GPUs
)

# High-end only
node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["h100"]  # H100 GPUs only
)
```

#### `exclude: Optional[List[str]] = None`

List of GPU types to exclude from selection. These GPU types will not be used even if available.

**Examples:**

```python
# Avoid specific GPU types
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    exclude=["t4"]  # Exclude T4 GPUs
)

# Cost optimization - exclude high-end GPUs
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    exclude=["h100", "a100-80gb"]  # Avoid expensive options
)

# Compatibility requirements
node_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40,
    exclude=["a10", "t4"]  # Exclude lower-tier GPUs
)
```

## Available GPU Types

### NVIDIA H100 Series
- **h100** - 80GB VRAM
  - Latest Hopper architecture
  - Best performance for large models
  - Highest cost tier

### NVIDIA A100 Series
- **a100** - 40GB VRAM
- **a100-80gb** - 80GB VRAM
  - Ampere architecture
  - Excellent for training and inference
  - High performance tier

### NVIDIA L40 Series
- **l40** - 48GB VRAM
  - Ada Lovelace architecture
  - Good balance of performance and cost
  - Optimized for inference

### Professional/Workstation GPUs
- **a6000** - 48GB VRAM
- **a5000** - 24GB VRAM
- **a4000** - 16GB VRAM
  - Professional-grade GPUs
  - Good for development and medium workloads

### Consumer/Entry GPUs
- **rtx4090** - 24GB VRAM
- **rtx3090** - 24GB VRAM
- **a10** - 24GB VRAM
- **t4** - 16GB VRAM
  - Cost-effective options
  - Suitable for smaller models

## Common Selection Patterns

### Cost-Optimized Selection

```python
# Small models - minimize cost
budget_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["t4", "a4000", "a10"]
)

# Medium models - balance cost/performance
balanced_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a5000", "rtx4090"],
    exclude=["h100", "a100-80gb"]
)
```

### Performance-Optimized Selection

```python
# Maximum performance
performance_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100-80gb"]
)

# High throughput serving
throughput_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=48,
    include=["l40", "a100"]
)
```

### Model-Specific Selection

```python
# 7B parameter models
llama_7b_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a5000", "rtx4090"]
)

# 13B parameter models
llama_13b_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=40,
    include=["l40", "a100", "a6000"]
)

# 70B parameter models
llama_70b_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100-80gb"]
)
```

## Best Practices

### 1. Right-Size Your Requirements

```python
# Don't over-provision
# Bad - wastes resources
oversized = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80
)

# Good - matches actual needs
rightsized = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)
```

### 2. Use Include/Exclude Wisely

```python
# Be specific when you have known requirements
specific_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48,
    include=["l40", "a6000"]  # Known compatible GPUs
)

# Exclude known incompatible GPUs
compatible_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    exclude=["t4"]  # Known to be too slow
)
```

### 3. Consider Multi-GPU for Large Models

```python
# Single large GPU vs multiple smaller GPUs
# Option 1: Single large GPU
single_gpu = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=80,
    include=["h100", "a100-80gb"]
)

# Option 2: Multiple smaller GPUs (often more available)
multi_gpu = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40,
    include=["a100", "l40"]
)
```

### 4. Development vs Production

```python
# Development - prioritize cost
dev_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["t4", "a4000"]
)

# Production - prioritize performance
prod_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=48,
    include=["l40", "a100"],
    exclude=["t4", "a4000"]
)
```

## Common Issues and Solutions

### Issue: "No available nodes match your requirements"

**Solution 1**: Broaden your requirements
```python
# Too restrictive
strict_selector = NodeSelector(
    gpu_count=8,
    min_vram_gb_per_gpu=80,
    include=["h100"]
)

# More flexible
flexible_selector = NodeSelector(
    gpu_count=4,
    min_vram_gb_per_gpu=48,
    include=["h100", "a100", "l40"]
)
```

**Solution 2**: Reduce GPU count
```python
# Instead of one 80GB GPU, try two 40GB GPUs
multi_gpu = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40
)
```

### Issue: "Out of memory" errors

**Solution**: Increase VRAM requirements
```python
# Increase min_vram_gb_per_gpu
higher_vram = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=48  # Increased from 24
)
```

## Integration Examples

### With Chute Definition

```python
from chutes import Chute, NodeSelector

# Define hardware requirements
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["l40", "a100"]
)

# Create chute with selector
chute = Chute(
    name="my-model-server",
    node_selector=node_selector
)
```

### With Template Functions

```python
from chutes.templates import build_vllm_chute
from chutes.chute import NodeSelector

# VLLM with specific GPU requirements
chute = build_vllm_chute(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24,
        include=["l40", "a5000"]
    )
)
```

### Dynamic Selection Based on Model

```python
def get_node_selector(model_size: str) -> NodeSelector:
    """Get appropriate NodeSelector based on model size."""
    
    if model_size == "small":  # < 3B parameters
        return NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=16
        )
    elif model_size == "medium":  # 7-13B parameters
        return NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=32,
            exclude=["t4"]
        )
    elif model_size == "large":  # 30-70B parameters
        return NodeSelector(
            gpu_count=2,
            min_vram_gb_per_gpu=48,
            include=["a100", "l40", "h100"]
        )
    else:  # > 70B parameters
        return NodeSelector(
            gpu_count=4,
            min_vram_gb_per_gpu=80,
            include=["h100", "a100-80gb"]
        )

# Usage
selector = get_node_selector("medium")
chute = Chute(
    name="llm-server",
    node_selector=selector
)
```

## Summary

The NodeSelector provides simple but powerful control over GPU hardware selection:

- **gpu_count**: Number of GPUs (1-8)
- **min_vram_gb_per_gpu**: Minimum VRAM per GPU (16-140 GB)
- **include**: Whitelist specific GPU types
- **exclude**: Blacklist specific GPU types

Always start with the minimum requirements for your workload and adjust based on performance needs and availability.