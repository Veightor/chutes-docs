# Custom Docker Images for Chutes

This guide demonstrates how to build custom Docker images for specialized use cases and advanced configurations in your Chutes applications.

## Overview

Custom images allow you to:

- **Pre-install Dependencies**: Include specific libraries, models, or tools
- **Optimize Performance**: Use custom Python versions or optimized libraries
- **Add System Tools**: Include CLI tools, databases, or other services
- **Custom Base Images**: Start from specialized base images (CUDA, Ubuntu, etc.)
- **Security Hardening**: Apply security configurations and patches

## Quick Examples

### Basic Custom Image

```python
from chutes.image import Image

# Simple custom image with additional packages
image = (
    Image(
        username="myuser",
        name="custom-nlp",
        tag="1.0.0",
        python_version="3.11"
    )
    .run_command("pip install transformers==4.35.0 torch==2.1.0 spacy==3.7.2")
    .run_command("python -m spacy download en_core_web_sm")
)
```

### GPU-Optimized Image

```python
from chutes.image import Image

# CUDA-optimized image for deep learning
image = (
    Image(
        username="myuser",
        name="gpu-ml",
        tag="cuda-12.1",
        base_image="nvidia/cuda:12.1-devel-ubuntu22.04",
        python_version="3.11"
    )
    .run_command("apt-get update && apt-get install -y git wget")
    .run_command("pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121")
    .run_command("pip install transformers>=4.35.0 accelerate>=0.24.0 bitsandbytes>=0.41.0")
)
```

## Advanced Configurations

### Multi-Stage Build

```python
from chutes.image import Image

# Multi-stage build for smaller final image
image = (
    Image(
        username="myuser",
        name="optimized-app",
        tag="slim",
        python_version="3.11"
    )
    # Build stage - install build dependencies
    .run_command("""
        apt-get update && apt-get install -y \\
        build-essential \\
        cmake \\
        git \\
        wget
    """)
    .run_command("pip install torch==2.1.0 transformers==4.35.0 opencv-python==4.8.0.76")
    # Cleanup stage - remove build dependencies
    .run_command("""
        apt-get autoremove -y build-essential cmake && \\
        apt-get clean && \\
        rm -rf /var/lib/apt/lists/*
    """)
    .add("./app", "/app")
    .set_workdir("/app")
)
```

### Custom Base with Pre-trained Models

```python
from chutes.image import Image

# Include pre-downloaded models in the image
image = (
    Image(
        username="myuser",
        name="llm-server",
        tag="mistral-7b",
        base_image="python:3.11-slim"
    )
    .run_command("mkdir -p /models")
    .add("./models/mistral-7b-instruct", "/models/mistral-7b-instruct")
    .run_command("pip install vllm==0.2.5 transformers==4.35.0 torch==2.1.0")
    .with_env("MODEL_PATH", "/models/mistral-7b-instruct")
    .with_env("CUDA_VISIBLE_DEVICES", "0")
)
```

### Database Integration

```python
from chutes.image import Image

# Image with PostgreSQL and Redis
image = (
    Image(
        username="myuser",
        name="full-stack-ai",
        tag="latest",
        base_image="ubuntu:22.04"
    )
    .run_command("""
        apt-get update && apt-get install -y \\
        python3.11 \\
        python3.11-pip \\
        postgresql-14 \\
        redis-server \\
        supervisor
    """)
    .run_command("pip install fastapi==0.104.1 uvicorn==0.24.0 psycopg2-binary==2.9.7 redis==5.0.0 sqlalchemy==2.0.23")
    .add("./config/supervisor.conf", "/etc/supervisor/conf.d/")
    .add("./app", "/app")
    .set_workdir("/app")
)
```

## Specialized Use Cases

### Computer Vision Pipeline

```python
from chutes.image import Image

# OpenCV + deep learning for computer vision
image = (
    Image(
        username="myuser",
        name="cv-pipeline",
        tag="opencv-4.8",
        python_version="3.11"
    )
    .run_command("""
        apt-get update && apt-get install -y \\
        libopencv-dev \\
        libglib2.0-0 \\
        libsm6 \\
        libxext6 \\
        libxrender-dev \\
        libgomp1 \\
        libglib2.0-0
    """)
    .run_command("pip install opencv-python==4.8.0.76 opencv-contrib-python==4.8.0.76 pillow==10.0.1 numpy==1.24.3 scikit-image==0.21.0 ultralytics==8.0.206")
    .add("./models/yolo", "/app/models/yolo")
    .add("./utils", "/app/utils")
)
```

### Audio Processing

```python
from chutes.image import Image

# Specialized audio processing environment
image = (
    Image(
        username="myuser",
        name="audio-ml",
        tag="latest",
        python_version="3.11"
    )
    .run_command("""
        apt-get update && apt-get install -y \\
        ffmpeg \\
        libsndfile1 \\
        libsndfile1-dev \\
        portaudio19-dev
    """)
    .run_command("pip install librosa==0.10.1 soundfile==0.12.1 pyaudio==0.2.11 pydub==0.25.1 whisper==1.1.10 torch==2.1.0 torchaudio==2.1.0")
    .add("./audio_models", "/app/models")
)
```

### Scientific Computing

```python
from chutes.image import Image

# Scientific Python stack with CUDA support
image = (
    Image(
        username="myuser",
        name="scientific-gpu",
        tag="cuda-scipy",
        base_image="nvidia/cuda:12.1-devel-ubuntu22.04"
    )
    .run_command("""
        apt-get update && apt-get install -y \\
        python3.11 \\
        python3.11-pip \\
        libhdf5-dev \\
        libnetcdf-dev \\
        gfortran
    """)
    .run_command("pip install numpy==1.24.3 scipy==1.11.4 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 jupyter==1.0.0 cupy-cuda12x==12.3.0 numba==0.58.1")
)
```

## Performance Optimization

### Layer Caching Strategy

```python
from chutes.image import Image

# Optimize layer caching for faster builds
image = (
    Image(
        username="myuser",
        name="cached-build",
        tag="optimized"
    )
    # 1. Install system dependencies first (rarely change)
    .run_command("apt-get update && apt-get install -y git wget")

    # 2. Install stable Python packages next
    .run_command("pip install numpy==1.24.3 pandas==2.0.3 requests==2.31.0")

    # 3. Install ML frameworks (change occasionally)
    .run_command("pip install torch==2.1.0 transformers==4.35.0")

    # 4. Copy application code last (changes frequently)
    .add("./src", "/app/src")
    .add("requirements-dev.txt", "/app/")
    .run_command("pip install -r /app/requirements-dev.txt")
)
```

### Minimizing Image Size

```python
from chutes.image import Image

# Minimal production image
image = (
    Image(
        username="myuser",
        name="minimal-prod",
        tag="slim",
        base_image="python:3.11-slim"
    )
    # Install only runtime dependencies
    .run_command("""
        apt-get update && \\
        apt-get install -y --no-install-recommends \\
        libgomp1 && \\
        apt-get clean && \\
        rm -rf /var/lib/apt/lists/*
    """)
    # Use --no-deps and specific versions
    .run_command("pip install torch==2.1.0+cpu transformers==4.35.0 --extra-index-url https://download.pytorch.org/whl/cpu")
    # Remove unnecessary files
    .run_command("""
        find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \\
        find /usr/local/lib/python3.11/site-packages -name "__pycache__" -delete
    """)
)
```

## Security Best Practices

### Secure Base Configuration

```python
from chutes.image import Image

# Security-hardened image
image = (
    Image(
        username="myuser",
        name="secure-app",
        tag="hardened",
        python_version="3.11"
    )
    # Create non-root user
    .run_command("""
        groupadd -r appuser && \\
        useradd -r -g appuser -d /app -s /sbin/nologin appuser
    """)
    # Install security updates
    .run_command("""
        apt-get update && \\
        apt-get upgrade -y && \\
        apt-get install -y --no-install-recommends \\
        ca-certificates && \\
        apt-get clean
    """)
    # Set up application directory
    .run_command("mkdir -p /app && chown -R appuser:appuser /app")
    .add("./app", "/app")
    .run_command("chown -R appuser:appuser /app")
    .set_workdir("/app")
    .set_user("appuser")
)
```

### Environment Variables Management

```python
from chutes.image import Image

# Secure environment setup
image = (
    Image(
        username="myuser",
        name="secure-env",
        tag="latest"
    )
    .with_env("PYTHONUNBUFFERED", "1")
    .with_env("PYTHONHASHSEED", "random")
    .with_env("PIP_NO_CACHE_DIR", "off")
    .with_env("PIP_DISABLE_PIP_VERSION_CHECK", "on")
    # Security settings
    .with_env("PYTHONDONTWRITEBYTECODE", "1")
    .with_env("PYTHONASYNCIODEBUG", "0")
)
```

## Integration Examples

### Using Custom Images in Chutes

```python
from chutes.chute import Chute, NodeSelector

# Deploy with custom image
chute = Chute(
    username="myuser",
    name="custom-ml-service",
    image=image,  # Your custom image from above
    entry_file="app.py",
    entry_point="run",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    timeout_seconds=300,
    concurrency=4
)

result = chute.deploy()
print(f"Deployed with custom image: {result}")
```

### Multi-Environment Deployment

```python
# Development image
dev_image = Image(
    username="myuser",
    name="ml-app",
    tag="dev"
).run_command("pip install pytest black flake8")

# Production image
prod_image = Image(
    username="myuser",
    name="ml-app",
    tag="prod"
).run_command("pip install gunicorn prometheus-client")

# Use different images per environment
if environment == "development":
    chute = Chute(name="ml-dev", image=dev_image, ...)
else:
    chute = Chute(name="ml-prod", image=prod_image, ...)
```

## Troubleshooting

### Common Issues

**Build Failures:**

```python
# Fix: Use explicit package versions
.run_command("pip install torch==2.1.0 numpy==1.24.3")  # Pin exact versions
```

**Large Image Sizes:**

```python
# Fix: Multi-stage builds and cleanup
.run_command("""
    apt-get update && apt-get install -y build-essential && \\
    pip install package && \\
    apt-get remove -y build-essential && \\
    apt-get autoremove -y && \\
    rm -rf /var/lib/apt/lists/*
""")
```

**Permission Issues:**

```python
# Fix: Set proper ownership
.add("./app", "/app")
.run_command("chown -R appuser:appuser /app")
```

### Debugging Images

```python
# Add debugging tools during development
debug_image = (
    base_image
    .run_command("pip install ipdb pdb++ memory-profiler")
    .run_command("apt-get install -y htop curl")
)
```

## Next Steps

- **[Performance Guide](../guides/performance)** - Optimize your custom images
- **[Best Practices](../guides/best-practices)** - Production deployment patterns
- **[Security Guide](../guides/security)** - Secure your applications
- **[Template Images](../templates/)** - Pre-built optimized images

For more complex configurations and enterprise use cases, see the [Advanced Docker Guide](../guides/advanced-docker).
