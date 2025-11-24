# Images (Docker Containers)

**Images** in Chutes define the Docker environment where your AI applications run. You can use pre-built images or create custom ones with a fluent Python API that generates optimized Dockerfiles.

## What is an Image?

An Image is a Docker container definition that includes:

- 🐧 **Base operating system** (usually Ubuntu with CUDA)
- 🐍 **Python environment** and packages
- 🧠 **AI frameworks** (PyTorch, TensorFlow, etc.)
- 📦 **System dependencies** and tools
- ⚙️ **Environment variables** and configuration
- 👤 **User setup** and permissions

## Using Pre-built Images

### Popular Base Images

```python
# NVIDIA CUDA images
"nvidia/cuda:12.2-devel-ubuntu22.04"
"nvidia/cuda:11.8-runtime-ubuntu20.04"

# Chutes optimized images
"chutes/cuda-python:12.2-py311"
"chutes/pytorch:2.1-cuda12.2"
"chutes/tensorflow:2.13-cuda11.8"

# Specialized AI framework images
"pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
"tensorflow/tensorflow:2.13.0-gpu"
```

### Using String References

```python
from chutes.chute import Chute

chute = Chute(
    username="myuser",
    name="my-chute",
    image="nvidia/cuda:12.2-devel-ubuntu22.04"  # Simple string reference
)
```

## Building Custom Images

### Basic Custom Image

```python
from chutes.image import Image

image = (
    Image(username="myuser", name="text-analyzer", tag="1.0")
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install torch transformers accelerate")
    .with_env("MODEL_CACHE", "/app/models")
)
```

### Image Constructor Parameters

#### Required Parameters

```python
Image(
    username="myuser",      # Your Chutes username
    name="my-image",        # Image name (alphanumeric + hyphens)
    tag="1.0"              # Version tag
)
```

#### Full Example

```python
image = Image(
    username="myuser",
    name="advanced-nlp",
    tag="2.1.3",
    readme="Advanced NLP processing with multiple models"
)
```

## Image Building Methods

### Base Image Selection

#### `.from_base(base_image: str)`

Set the base Docker image:

```python
# CUDA development environment
.from_base("nvidia/cuda:12.2-devel-ubuntu22.04")

# Lightweight runtime
.from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")

# Pre-built PyTorch
.from_base("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
```

### Python Environment

#### `.with_python(version: str)`

Install a specific Python version:

```python
.with_python("3.11")    # Python 3.11 (recommended)
.with_python("3.10")    # Python 3.10
.with_python("3.9")     # Python 3.9
```

#### Installing Python Packages

Use `run_command()` to install Python packages:

```python
# Individual packages
.run_command("pip install torch transformers numpy")

# With versions
.run_command("pip install torch==2.1.0 transformers>=4.21.0")

# From requirements file
.run_command("pip install -r requirements.txt")
```

#### Installing Conda Packages

Use `run_command()` to install packages via conda:

```python
.run_command("conda install pytorch torchvision torchaudio")
.run_command("conda install cudatoolkit=11.8 numpy scipy")
```

### System Commands

#### `.run_command(command: str)`

Execute arbitrary shell commands:

```python
# Install system packages
.run_command("apt-get update && apt-get install -y git curl wget")

# Download models
.run_command("wget https://example.com/model.bin -O /app/model.bin")

# Set up directories
.run_command("mkdir -p /app/models /app/data /app/logs")

# Compile native extensions
.run_command("cd /app && python setup.py build_ext --inplace")
```

### Environment Variables

#### `.with_env(key: str, value: str)`

Set environment variables:

```python
.with_env("CUDA_VISIBLE_DEVICES", "0")
.with_env("TRANSFORMERS_CACHE", "/app/cache")
.with_env("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
.with_env("MODEL_PATH", "/app/models/my-model")
```

### File Operations

#### `.add(*args, **kwargs)`

Add files to the image:

```python
# Add files to the image
.add("config.json", "/app/config.json")

# Add directories
.add("models/", "/app/models/")

# Add requirements file
.add("requirements.txt", "/app/requirements.txt")
```

### User Management

#### `.set_user(user: str)`

Set the user for the container:

```python
# Set user
.set_user("appuser")

# Set user for chutes
.set_user("chutes")
```

#### `.set_workdir(directory: str)`

Set the working directory:

```python
.set_workdir("/app")
.set_workdir("/workspace/myproject")
```

## Complete Example

```python
from chutes.image import Image

# Build a comprehensive NLP processing image
image = (
    Image(
        username="myuser",
        name="nlp-suite",
        tag="1.2.0",
        description="Complete NLP processing suite with multiple models"
    )
    # Start with CUDA base
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")

    # Install system dependencies
    .run_command("""
        apt-get update && apt-get install -y \\
        git curl wget unzip \\
        build-essential \\
        ffmpeg \\
        && rm -rf /var/lib/apt/lists/*
    """)

    # Set up Python
    .with_python("3.11")

    # Install core ML packages
    .run_command("""
        pip install \\
        torch==2.1.0 \\
        torchvision==0.16.0 \\
        torchaudio==2.1.0 \\
        transformers>=4.30.0 \\
        accelerate>=0.20.0 \\
        datasets>=2.12.0 \\
        tokenizers>=0.13.0
    """)

    # Install additional NLP tools
    .run_command("""
        pip install \\
        spacy>=3.6.0 \\
        nltk>=3.8 \\
        scikit-learn>=1.3.0 \\
        pandas>=2.0.0 \\
        numpy>=1.24.0
    """)

    # Set up directories
    .run_command("mkdir -p /app/models /app/data /app/cache /app/logs")

    # Add application files
    .add("requirements.txt", "/app/requirements.txt")
    .add("src/", "/app/src/")
    .add("config/", "/app/config/")

    # Set environment variables
    .with_env("TRANSFORMERS_CACHE", "/app/cache")
    .with_env("HF_HOME", "/app/cache")
    .with_env("TORCH_HOME", "/app/cache/torch")
    .with_env("PYTHONPATH", "/app/src")

    # Download spaCy models
    .run_command("python -m spacy download en_core_web_sm")
    .run_command("python -m spacy download en_core_web_lg")

    # Download NLTK data
    .run_command("""
        python -c "
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        "
    """)

    # Set working directory and user
    .set_workdir("/app")
    .set_user("appuser")
)
```

## Advanced Features

### Multi-stage Builds

```python
# Build stage for compiling
build_image = (
    Image(username="myuser", name="builder", tag="temp")
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install cython numpy")
    .copy_file("src/", "/build/src/")
    .run_command("cd /build && python setup.py build_ext")
)

# Production stage with compiled artifacts
production_image = (
    Image(username="myuser", name="production", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .add("/build/dist/", "/app/")
    .run_command("pip install torch transformers")
)
```

### Conditional Building

```python
def build_image_for_gpu(gpu_type: str) -> Image:
    image = (
        Image(username="myuser", name=f"model-{gpu_type}", tag="1.0")
        .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
        .with_python("3.11")
    )

    if gpu_type == "a100":
        # Optimize for A100
        image = image.with_env("TORCH_CUDA_ARCH_LIST", "8.0")
    elif gpu_type == "v100":
        # Optimize for V100
        image = image.with_env("TORCH_CUDA_ARCH_LIST", "7.0")

    return image.run_command("pip install torch transformers")
```

### Template Images

```python
def create_pytorch_image(username: str, name: str, pytorch_version: str = "2.1.0") -> Image:
    """Template for PyTorch-based images"""
    return (
        Image(username=username, name=name, tag=pytorch_version)
        .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
        .with_python("3.11")
        .run_command(f"pip install torch=={pytorch_version}")
        .run_command("pip install torchvision torchaudio")
        .with_env("TORCH_CUDA_ARCH_LIST", "7.0;8.0;8.6")
        .set_workdir("/app")
    )

# Use the template
my_image = create_pytorch_image("myuser", "my-pytorch-app")
```

## Image Building Process

### Local Building

```bash
# Build image locally
chutes build my_chute:chute --wait

# Build with custom tag
chutes build my_chute:chute --tag custom-v1.0

# Build without cache
chutes build my_chute:chute --no-cache
```

### Remote Building

Images are built on Chutes infrastructure with:

- 🚀 **Fast build times** with optimized caching
- 🔒 **Secure environment** with isolated builds
- 📦 **Automatic registry** management
- 🏗️ **Multi-architecture** support

### Build Optimization

```python
# Layer caching - put stable operations first
image = (
    Image(username="myuser", name="optimized", tag="1.0")
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")

    # System packages (rarely change)
    .run_command("apt-get update && apt-get install -y git curl")

    # Python installation (stable)
    .with_python("3.11")

    # Core dependencies (change less frequently)
    .run_command("pip install torch==2.1.0 transformers==4.30.0")

    # Application-specific packages (change more frequently)
    .run_command("pip install -r requirements.txt")

    # Application code (changes most frequently)
    .add("src/", "/app/src/")
)
```

## Best Practices

### 1. Layer Optimization

```python
# Good: Group related operations
.run_command("""
    apt-get update && \\
    apt-get install -y git curl wget && \\
    rm -rf /var/lib/apt/lists/*
""")

# Bad: Separate operations create more layers
.run_command("apt-get update")
.run_command("apt-get install -y git")
.run_command("apt-get install -y curl")
```

### 2. Security

```python
# Use specific versions
.run_command("pip install torch==2.1.0 transformers==4.30.0")

# Create non-root user
.set_user("appuser")

# Clean up package caches
.run_command("apt-get clean && rm -rf /var/lib/apt/lists/*")
```

### 3. Size Optimization

```python
# Combine operations to reduce layers
.run_command("""
    pip install torch transformers && \\
    pip cache purge && \\
    rm -rf ~/.cache/pip
""")

# Add only what you need
.add("src/", "/app/src/")  # Only add what you need
```

### 4. Environment Consistency

```python
# Pin all versions
.with_python("3.11.5")
.run_command("pip install torch==2.1.0+cu121 transformers==4.30.2")

# Set explicit environment
.with_env("PYTHONPATH", "/app/src")
.with_env("CUDA_VISIBLE_DEVICES", "0")
```

## Common Patterns

### AI Framework Setup

```python
# PyTorch with CUDA
pytorch_image = (
    Image(username="myuser", name="pytorch-app", tag="1.0")
    .from_base("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
    .run_command("pip install transformers accelerate datasets")
    .with_env("TORCH_CUDA_ARCH_LIST", "7.0;8.0;8.6")
)

# TensorFlow with CUDA
tensorflow_image = (
    Image(username="myuser", name="tensorflow-app", tag="1.0")
    .from_base("tensorflow/tensorflow:2.13.0-gpu")
    .run_command("pip install tensorflow-datasets tensorflow-hub")
    .with_env("TF_FORCE_GPU_ALLOW_GROWTH", "true")
)
```

### Model Downloading

```python
model_image = (
    Image(username="myuser", name="model-app", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install transformers torch")

    # Pre-download models during build
    .run_command("""
        python -c "
        from transformers import AutoModel, AutoTokenizer
        AutoModel.from_pretrained('bert-base-uncased')
        AutoTokenizer.from_pretrained('bert-base-uncased')
        "
    """)
    .with_env("TRANSFORMERS_CACHE", "/app/cache")
)
```

## Next Steps

- **[Chutes](/docs/core-concepts/chutes)** - Learn how to use images in Chutes
- **[Node Selection](/docs/core-concepts/node-selection)** - Hardware requirements
- **[Custom Image Building Guide](/docs/guides/custom-images)** - Advanced image building
- **[Template Images](/docs/core-concepts/templates)** - Pre-built image templates
