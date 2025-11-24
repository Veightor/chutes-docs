# Image API Reference

The `Image` class is used to build custom Docker images for Chutes applications. This reference covers all methods, configuration options, and best practices for creating optimized container images.

## Class Definition

```python
from chutes.image import Image

image = Image(
    username: str,
    name: str,
    tag: str = "latest",
    readme: Optional[str] = None
)
```

## Constructor Parameters

### Required Parameters

#### `username: str`

The username or organization name for the image.

**Example:**

```python
image = Image(username="mycompany", name="custom-ai")
```

**Rules:**

- Must be lowercase alphanumeric with hyphens
- Cannot start or end with hyphen
- Maximum 63 characters
- Should match your Chutes username

#### `name: str`

The name of the Docker image.

**Example:**

```python
image = Image(username="mycompany", name="text-processor")
```

**Rules:**

- Must be lowercase alphanumeric with hyphens
- Cannot start or end with hyphen
- Maximum 63 characters
- Should be descriptive of the image purpose

### Optional Parameters

#### `tag: str = "latest"`

Version tag for the image.

**Examples:**

```python
# Default latest tag
image = Image(username="mycompany", name="ai-model")

# Specific version tag
image = Image(username="mycompany", name="ai-model", tag="1.0.0")

# Development tag
image = Image(username="mycompany", name="ai-model", tag="dev")

# Feature branch tag
image = Image(username="mycompany", name="ai-model", tag="feature-new-model")
```

**Best Practices:**

- Use semantic versioning (1.0.0, 1.1.0, etc.)
- Use descriptive tags for different environments
- Avoid using "latest" in production

#### `readme: Optional[str] = None`

Documentation for the image in Markdown format.

**Example:**

```python
readme = """
# Custom AI Processing Image

This image contains optimized libraries for AI text processing.

## Features
- PyTorch 2.0 with CUDA support
- Transformers library
- Custom preprocessing tools
- Optimized for GPU inference

## Usage
This image is designed for text generation workloads.
"""

image = Image(
    username="mycompany",
    name="ai-processor",
    tag="1.0.0",
    readme=readme
)
```

## Core Methods

### Base Image Configuration

#### `.from_base(base_image: str)`

Set the base Docker image to build from.

**Signature:**

```python
def from_base(self, base_image: str) -> Image
```

**Examples:**

```python
# Python base images
image = Image("myuser", "myapp").from_base("python:3.11-slim")
image = Image("myuser", "myapp").from_base("python:3.11-bullseye")

# Ubuntu base images
image = Image("myuser", "myapp").from_base("ubuntu:22.04")
image = Image("myuser", "myapp").from_base("ubuntu:20.04")

# NVIDIA CUDA base images
image = Image("myuser", "myapp").from_base("nvidia/cuda:11.8-devel-ubuntu22.04")
image = Image("myuser", "myapp").from_base("nvidia/cuda:12.1-runtime-ubuntu22.04")

# Specialized base images
image = Image("myuser", "myapp").from_base("tensorflow/tensorflow:2.13.0-gpu")
image = Image("myuser", "myapp").from_base("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime")

# Chutes base images
image = Image("myuser", "myapp").from_base("parachutes/base-python:3.11.9")
```

**Choosing Base Images:**

- **python:3.11-slim**: Lightweight Python, good for CPU workloads
- **nvidia/cuda:\***: For GPU-accelerated applications
- **ubuntu:22.04**: When you need full system control
- **tensorflow/tensorflow:\***: Pre-configured TensorFlow environment
- **pytorch/pytorch:\***: Pre-configured PyTorch environment

### Package Installation

#### `.run_command(command: str)`

Execute shell commands during image build.

**Signature:**

```python
def run_command(self, command: str) -> Image
```

**Examples:**

```python
# Install system packages
image = (
    Image("myuser", "myapp")
    .from_base("ubuntu:22.04")
    .run_command("apt update && apt install -y python3 python3-pip")
)

# Install Python packages
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .run_command("pip install torch torchvision torchaudio")
)

# Multiple commands in one call
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .run_command("""
        apt update &&
        apt install -y git curl &&
        pip install --upgrade pip &&
        pip install numpy pandas scikit-learn
    """)
)

# Install from requirements file
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .add("requirements.txt", "/tmp/requirements.txt")
    .run_command("pip install -r /tmp/requirements.txt")
)
```

**Common Patterns:**

```python
# System dependencies for AI/ML
image = image.run_command("""
    apt update && apt install -y
    build-essential
    git
    curl
    wget
    ffmpeg
    libsm6
    libxext6
    libfontconfig1
    libxrender1
""")

# Python ML stack
image = image.run_command("""
    pip install
    torch
    transformers
    accelerate
    datasets
    tokenizers
    numpy
    pandas
    scikit-learn
    pillow
    opencv-python
""")

# Clean up after installation
image = image.run_command("""
    apt autoremove -y &&
    apt clean &&
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
""")
```

### File Operations

#### `.add(*args, **kwargs)`

Add files from the build context to the image.

**Signature:**

```python
def add(self, *args, **kwargs) -> Image
```

**Examples:**

```python
# Add single file
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .add("requirements.txt", "/app/requirements.txt")
)

# Add directory
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .add("src/", "/app/src/")
)

# Add with different name
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .add("config.yaml", "/app/production-config.yaml")
)

# Add multiple files
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .add("requirements.txt", "/app/requirements.txt")
    .add("setup.py", "/app/setup.py")
    .add("src/", "/app/src/")
)
```

**Best Practices:**

```python
# Add requirements first for better caching
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .add("requirements.txt", "/tmp/requirements.txt")  # Add early
    .run_command("pip install -r /tmp/requirements.txt")     # Install deps
    .add("src/", "/app/src/")                          # Add code last
)
```

**Note:** Multi-stage builds are not directly supported by the current Image API. Use external build tools for multi-stage builds.

### Environment Configuration

#### `.with_env(key: str, value: str)`

Set environment variables in the image.

**Signature:**

```python
def with_env(self, key: str, value: str) -> Image
```

**Examples:**

```python
# Basic environment variables
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .with_env("PYTHONPATH", "/app")
    .with_env("PYTHONUNBUFFERED", "1")
)

# GPU configuration
image = (
    Image("myuser", "gpu-app")
    .from_base("nvidia/cuda:11.8-runtime")
    .with_env("CUDA_VISIBLE_DEVICES", "0")
    .with_env("NVIDIA_VISIBLE_DEVICES", "all")
    .with_env("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
)

# Application configuration
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .with_env("APP_ENV", "production")
    .with_env("LOG_LEVEL", "INFO")
    .with_env("MAX_WORKERS", "4")
)

# Model configuration
image = (
    Image("myuser", "ai-app")
    .from_base("python:3.11-slim")
    .with_env("TRANSFORMERS_CACHE", "/opt/models")
    .with_env("HF_HOME", "/opt/huggingface")
    .with_env("TORCH_HOME", "/opt/torch")
)
```

**Common Environment Variables:**

```python
# Python optimization
image = image.with_env("PYTHONOPTIMIZE", "2")          # Enable optimizations
image = image.with_env("PYTHONDONTWRITEBYTECODE", "1") # Don't write .pyc files
image = image.with_env("PYTHONUNBUFFERED", "1")        # Unbuffered output

# PyTorch optimizations
image = image.with_env("TORCH_BACKENDS_CUDNN_BENCHMARK", "1")
image = image.with_env("TORCH_BACKENDS_CUDNN_DETERMINISTIC", "0")

# Memory management
image = image.with_env("MALLOC_ARENA_MAX", "4")        # Reduce memory fragmentation
```

#### `.set_workdir(directory: str)`

Set the working directory for the container.

**Signature:**

```python
def set_workdir(self, directory: str) -> Image
```

**Examples:**

```python
# Set working directory
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .set_workdir("/app")
    .add(".", "/app")
)

# Multiple working directories for different stages
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .set_workdir("/tmp")
    .add("requirements.txt", "requirements.txt")
    .run_command("pip install -r requirements.txt")
    .set_workdir("/app")
    .add("src/", ".")
)
```

#### `.set_user(user: str)`

Set the user for running commands and the container.

**Signature:**

```python
def set_user(self, user: str) -> Image
```

**Examples:**

```python
# Create and use non-root user
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .run_command("useradd -m -u 1000 appuser")
    .run_command("mkdir -p /app && chown appuser:appuser /app")
    .set_user("appuser")
    .set_workdir("/app")
)

# Use existing user
image = (
    Image("myuser", "myapp")
    .from_base("ubuntu:22.04")
    .set_user("nobody")
)

# Switch between users
image = (
    Image("myuser", "myapp")
    .from_base("ubuntu:22.04")
    .run_command("apt update && apt install -y python3")  # As root
    .run_command("useradd -m appuser")                     # As root
    .set_user("appuser")                                  # Switch to appuser
    .run_command("whoami")                                 # As appuser
)
```

### Container Configuration

**Note:** Port exposure is handled automatically by the Chutes platform.

#### `.with_entrypoint(entrypoint: List[str])`

Set the container entrypoint.

**Signature:**

```python
def with_entrypoint(self, entrypoint: List[str]) -> Image
```

**Examples:**

```python
# Python application entrypoint
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .with_entrypoint(["python", "-m", "myapp"])
)

# Shell script entrypoint
image = (
    Image("myuser", "myapp")
    .from_base("ubuntu:22.04")
    .add("entrypoint.sh", "/entrypoint.sh")
    .run_command("chmod +x /entrypoint.sh")
    .with_entrypoint(["/entrypoint.sh"])
)

# Complex entrypoint with arguments
image = (
    Image("myuser", "myapp")
    .from_base("python:3.11-slim")
    .with_entrypoint([
        "python",
        "-u",           # Unbuffered output
        "-m", "uvicorn",
        "myapp:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
)
```

**Note:** Default commands are handled by the Chutes platform and chute definitions.

**Note:** Image labels and metadata are handled by the Chutes platform.

**Note:** Health checks are handled by the Chutes platform.

## Advanced Patterns

### Multi-Stage Builds

```python
# Build stage
build_stage = (
    Image("myuser", "builder", "latest")
    .from_base("python:3.11")
    .run_command("apt update && apt install -y build-essential git")
    .add("requirements.txt", "/tmp/requirements.txt")
    .run_command("pip install -r /tmp/requirements.txt --target /opt/python-packages")
    .add("src/", "/tmp/src/")
    .run_command("cd /tmp/src && python setup.py build_ext --inplace")
)

# Production stage
production_stage = (
    Image("myuser", "myapp", "1.0.0")
    .from_base("python:3.11-slim")
    .run_command("apt update && apt install -y --no-install-recommends libgcc-s1")
    # Note: Multi-stage builds not supported in current API
    .with_env("PYTHONPATH", "/opt/python-packages")
    .set_workdir("/app")
    .with_user("1000:1000")
    .with_cmd(["python", "main.py"])
)
```

### Optimized AI/ML Images

```python
def create_pytorch_image(model_name: str, version: str) -> Image:
    """Create optimized PyTorch image for specific model."""

    return (
        Image("myuser", f"pytorch-{model_name}", version)
        .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")

        # System dependencies
        .run_command("""
            apt update && apt install -y
            python3 python3-pip python3-dev
            build-essential git curl wget
            libsm6 libxext6 libfontconfig1 libxrender1
        """)

        # Python environment
        .run_command("pip3 install --upgrade pip setuptools wheel")

        # PyTorch with CUDA
        .run_command("""
            pip3 install torch torchvision torchaudio
            --index-url https://download.pytorch.org/whl/cu118
        """)

        # ML libraries
        .run_command("""
            pip3 install
            transformers[torch]
            accelerate
            datasets
            tokenizers
            optimum
        """)

        # Performance libraries
        .run_command("pip3 install flash-attn --no-build-isolation")
        .run_command("pip3 install xformers")

        # Environment optimization
        .with_env("PYTHONOPTIMIZE", "2")
        .with_env("TORCH_BACKENDS_CUDNN_BENCHMARK", "1")
        .with_env("TRANSFORMERS_CACHE", "/opt/models")
        .with_env("HF_HOME", "/opt/huggingface")

        # Setup directories
        .run_command("mkdir -p /opt/models /opt/huggingface /app")
        .run_command("chown -R 1000:1000 /opt/models /opt/huggingface /app")

        # Non-root user
        .set_user("1000:1000")
        .set_workdir("/app")

        # Cleanup
        .run_command("""
            apt autoremove -y &&
            apt clean &&
            rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/pip
        """)
    )

# Usage
pytorch_image = create_pytorch_image("gpt2", "1.0.0")
```

### Layered Configuration

```python
def create_base_ai_image() -> Image:
    """Create base image for AI applications."""

    return (
        Image("myuser", "ai-base", "1.0.0")
        .from_base("nvidia/cuda:11.8-runtime-ubuntu22.04")
        .run_command("apt update && apt install -y python3 python3-pip")
        .run_command("pip3 install --upgrade pip")
        .with_env("PYTHONUNBUFFERED", "1")
        .with_env("PYTHONDONTWRITEBYTECODE", "1")
    )

def create_ml_image(base_image: Image) -> Image:
    """Add ML libraries to base image."""

    return (
        base_image
        .run_command("pip3 install numpy pandas scikit-learn")
        .run_command("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        .with_env("TORCH_HOME", "/opt/torch")
    )

def create_nlp_image(ml_image: Image) -> Image:
    """Add NLP libraries to ML image."""

    return (
        ml_image
        .run_command("pip3 install transformers tokenizers datasets")
        .run_command("pip3 install spacy nltk")
        .with_env("TRANSFORMERS_CACHE", "/opt/models")
    )

# Build layered images
base = create_base_ai_image()
ml = create_ml_image(base)
nlp = create_nlp_image(ml)
```

## Performance Optimization

### Build Optimization

```python
# Optimized build order
def create_optimized_image() -> Image:
    """Create image with optimized layer caching."""

    return (
        Image("myuser", "optimized-app", "1.0.0")
        .from_base("python:3.11-slim")

        # System packages (changes rarely)
        .run_command("""
            apt update && apt install -y --no-install-recommends
            build-essential git curl
            && rm -rf /var/lib/apt/lists/*
        """)

        # Python dependencies (changes occasionally)
        .add("requirements.txt", "/tmp/requirements.txt")
        .run_command("pip install --no-cache-dir -r /tmp/requirements.txt")

        # Application code (changes frequently)
        .add("src/", "/app/src/")
        .set_workdir("/app")

        # Runtime configuration
        .with_env("PYTHONPATH", "/app")
        .set_user("1000:1000")
        .expose_port(8000)
        .with_cmd(["python", "-m", "src.main"])
    )
```

### Size Optimization

```python
def create_minimal_image() -> Image:
    """Create minimal size image."""

    return (
        Image("myuser", "minimal-app", "1.0.0")
        .from_base("python:3.11-alpine")  # Smaller base

        # Install only essential packages
        .run_command("""
            apk add --no-cache
            gcc musl-dev linux-headers
            && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
            && apk del gcc musl-dev linux-headers
        """)

        # Copy only necessary files
        .add("src/main.py", "/app/main.py")
        .add("requirements-minimal.txt", "/tmp/requirements.txt")
        .run_command("pip install --no-cache-dir -r /tmp/requirements.txt")

        # Remove temporary files
        .run_command("rm -rf /tmp/* /var/cache/apk/*")

        # Minimal runtime
        .set_workdir("/app")
        .set_user("65534:65534")  # nobody user
        .with_cmd(["python", "main.py"])
    )
```

## Testing Images

### Testing Image Builds

```python
import docker
import pytest

def test_image_builds_successfully():
    """Test that image builds without errors."""

    image = (
        Image("test", "myapp", "test")
        .from_base("python:3.11-slim")
        .run_command("pip install requests")
    )

    # This would build the image
    # Implementation depends on your build system
    assert image is not None

def test_image_has_correct_environment():
    """Test that image has correct environment variables."""

    image = (
        Image("test", "myapp", "test")
        .from_base("python:3.11-slim")
        .with_env("TEST_VAR", "test_value")
    )

    # Test environment configuration
    # Implementation depends on your testing framework
    pass

def test_image_security():
    """Test image security configurations."""

    image = (
        Image("test", "secure-app", "test")
        .from_base("python:3.11-slim")
        .set_user("1000:1000")  # Non-root user
    )

    # Test security settings
    # Implementation depends on your security scanning tools
    pass
```

## Best Practices

### Security Best Practices

1. **Use Non-Root Users**

   ```python
   image = (
       image
       .run_command("useradd -m -u 1000 appuser")
       .with_user("appuser")
   )
   ```

2. **Minimize Attack Surface**

   ```python
   image = (
       image
       .from_base("python:3.11-slim")  # Minimal base
       .run_command("apt remove --purge -y wget curl")  # Remove unnecessary tools
   )
   ```

3. **Keep Images Updated**
   ```python
   image = (
       image
       .run_command("apt update && apt upgrade -y")  # Update packages
   )
   ```

### Performance Best Practices

1. **Optimize Layer Caching**

   ```python
   # Copy requirements first
   image = (
       image
       .add("requirements.txt", "/tmp/requirements.txt")
       .run_command("pip install -r /tmp/requirements.txt")
       .add("src/", "/app/src/")  # Copy code last
   )
   ```

2. **Combine RUN Commands**

   ```python
   # Good: Single layer
   image = image.run_command("""
       apt update &&
       apt install -y python3 &&
       rm -rf /var/lib/apt/lists/*
   """)

   # Avoid: Multiple layers
   # image = image.run_command("apt update")
   # image = image.run_command("apt install -y python3")
   # image = image.run_command("rm -rf /var/lib/apt/lists/*")
   ```

3. **Clean Up in Same Layer**
   ```python
   image = image.run_command("""
       pip install large-package &&
       rm -rf ~/.cache/pip  # Clean up in same layer
   """)
   ```

### Maintainability Best Practices

1. **Use Descriptive Tags**

   ```python
   image = Image("myuser", "myapp", "v1.2.3-python3.11-cuda11.8")
   ```

2. **Add Comprehensive Labels**

   ```python
   image = (
       image
       .with_label("version", "1.0.0")
       .with_label("description", "AI text processing service")
       .with_label("maintainer", "team@company.com")
   )
   ```

3. **Document Complex Builds**

   ```python
   readme = """
   # Custom AI Image

   ## Base Image
   - nvidia/cuda:11.8-runtime-ubuntu22.04

   ## Installed Packages
   - PyTorch 2.0 with CUDA 11.8
   - Transformers library
   - Custom preprocessing tools

   ## Usage
   Designed for GPU-accelerated text generation.
   """

   image = Image("myuser", "ai-app", "1.0.0", readme=readme)
   ```

This comprehensive guide covers all aspects of the `Image` class for building optimized Docker images in Chutes applications.
