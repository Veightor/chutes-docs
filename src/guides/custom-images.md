# Custom Image Building

This guide covers advanced Docker image building techniques for Chutes, enabling you to create optimized, production-ready containers for AI applications with custom dependencies, performance tuning, and security considerations.

## Overview

Custom images in Chutes provide:

- **Full Control**: Complete control over the software stack
- **Optimization**: Fine-tuned performance for specific workloads
- **Custom Dependencies**: Any Python packages, system libraries, or tools
- **Reproducibility**: Versioned, immutable deployments
- **Caching**: Intelligent layer caching for fast rebuilds
- **Security**: Hardened containers with minimal attack surface

## Basic Image Building

### Simple Custom Image

```python
from chutes.image import Image

# Basic custom image
image = (
    Image(username="myuser", name="my-app", tag="1.0")
    .from_base("python:3.11-slim")
    .run_command("pip install numpy pandas scikit-learn")
    .with_workdir("/app")
)
```

### Fluent API Patterns

The Chutes Image class uses a fluent API for building complex Docker images:

```python
image = (
    Image(username="myuser", name="ai-pipeline", tag="2.1")
    .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")

    # System setup
    .run_command("apt update && apt install -y python3 python3-pip git curl")
    .run_command("apt install -y ffmpeg libsm6 libxext6")  # OpenCV dependencies

    # Python environment
    .run_command("pip3 install --upgrade pip setuptools wheel")
    .run_command("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    # AI libraries
    .run_command("pip3 install transformers accelerate")
    .run_command("pip3 install opencv-python pillow")
    .run_command("pip3 install fastapi uvicorn pydantic")

    # Environment configuration
    .with_env("PYTHONPATH", "/app")
    .with_env("CUDA_VISIBLE_DEVICES", "0")
    .with_workdir("/app")

    # User setup for security
    .run_command("useradd -m -u 1000 appuser")
    .run_command("chown -R appuser:appuser /app")
    .with_user("appuser")
)
```

## Advanced Image Building Patterns

### Multi-Stage Builds

Use multi-stage builds for smaller, more secure production images:

```python
# Build stage
build_image = (
    Image(username="myuser", name="ai-builder", tag="build")
    .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")
    .run_command("apt update && apt install -y python3 python3-pip git build-essential")
    .run_command("pip3 install --upgrade pip setuptools wheel")

    # Install build dependencies
    .run_command("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    .run_command("pip3 install transformers[torch]")
    .run_command("pip3 install accelerate bitsandbytes")

    # Compile custom CUDA kernels if needed
    .run_command("pip3 install flash-attn --no-build-isolation")
    .run_command("pip3 install apex --no-build-isolation")

    # Copy application code
    .copy_file("requirements.txt", "/tmp/requirements.txt")
    .run_command("pip3 install -r /tmp/requirements.txt")
)

# Production stage - smaller runtime image
production_image = (
    Image(username="myuser", name="ai-runtime", tag="1.0")
    .from_base("nvidia/cuda:11.8-runtime-ubuntu22.04")  # Runtime only, not devel
    .run_command("apt update && apt install -y python3 python3-pip")
    .run_command("rm -rf /var/lib/apt/lists/*")  # Clean up package cache

    # Note: copy_from_image not available - use external build process

    # Application setup
    .set_workdir("/app")
    .set_user("appuser")  # Non-root user
)
```

### GPU-Optimized Images

Build images optimized for different GPU architectures:

```python
def create_gpu_optimized_image(gpu_arch: str = "ampere"):
    """Create GPU-optimized image for specific architecture."""

    # Base images optimized for different GPU generations
    base_images = {
        "pascal": "nvidia/cuda:11.2-devel-ubuntu20.04",    # GTX 10xx, P100
        "volta": "nvidia/cuda:11.4-devel-ubuntu20.04",     # V100, Titan V
        "turing": "nvidia/cuda:11.6-devel-ubuntu20.04",    # RTX 20xx, T4
        "ampere": "nvidia/cuda:11.8-devel-ubuntu22.04",    # RTX 30xx, A100
        "ada": "nvidia/cuda:12.1-devel-ubuntu22.04",       # RTX 40xx
        "hopper": "nvidia/cuda:12.2-devel-ubuntu22.04",    # H100
    }

    # Architecture-specific optimizations
    torch_arch_flags = {
        "pascal": "6.0;6.1",
        "volta": "7.0",
        "turing": "7.5",
        "ampere": "8.0;8.6",
        "ada": "8.9",
        "hopper": "9.0"
    }

    base_image = base_images.get(gpu_arch, base_images["ampere"])
    arch_flags = torch_arch_flags.get(gpu_arch, "8.0;8.6")

    return (
        Image(username="myuser", name=f"gpu-{gpu_arch}", tag="1.0")
        .from_base(base_image)
        .with_env("TORCH_CUDA_ARCH_LIST", arch_flags)
        .with_env("CUDA_ARCHITECTURES", arch_flags.replace(";", " "))

        # Install optimized PyTorch
        .run_command("pip3 install --upgrade pip")
        .run_command("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

        # Compile architecture-specific kernels
        .run_command(f"pip3 install flash-attn --no-build-isolation")
        .run_command("pip3 install xformers")  # Memory-efficient attention

        # Install performance libraries
        .run_command("pip3 install triton")   # CUDA kernel compilation
        .run_command("pip3 install apex --no-build-isolation")  # Mixed precision
    )

# Usage
ampere_image = create_gpu_optimized_image("ampere")  # For A100, RTX 30xx
hopper_image = create_gpu_optimized_image("hopper")  # For H100
```

### AI Framework-Specific Images

Create specialized images for different AI frameworks:

```python
class AIFrameworkImages:
    """Collection of framework-specific image builders."""

    @staticmethod
    def pytorch_image(version: str = "2.1.0", cuda_version: str = "11.8"):
        """PyTorch optimized image."""
        return (
            Image(username="myuser", name="pytorch", tag=version)
            .from_base(f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04")
            .run_command("apt update && apt install -y python3 python3-pip")

            # Install PyTorch with CUDA support
            .run_command(f"pip3 install torch=={version} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}")

            # Performance optimizations
            .run_command("pip3 install accelerate")
            .run_command("pip3 install xformers")  # Memory-efficient transformers
            .run_command("pip3 install flash-attn --no-build-isolation")

            # Common ML libraries
            .run_command("pip3 install transformers datasets tokenizers")
            .run_command("pip3 install numpy scipy scikit-learn pandas")

            # Environment optimizations
            .with_env("TORCH_BACKENDS_CUDNN_BENCHMARK", "1")
            .with_env("TORCH_BACKENDS_CUDNN_DETERMINISTIC", "0")
        )

    @staticmethod
    def tensorflow_image(version: str = "2.13.0"):
        """TensorFlow optimized image."""
        return (
            Image(username="myuser", name="tensorflow", tag=version)
            .from_base("tensorflow/tensorflow:2.13.0-gpu")

            # Additional TF ecosystem
            .run_command("pip3 install tensorflow-hub tensorflow-datasets")
            .run_command("pip3 install tensorflow-probability")
            .run_command("pip3 install tensorboard")

            # Optimization libraries
            .run_command("pip3 install tf-keras-vis")  # Visualization
            .run_command("pip3 install tensorflow-model-optimization")  # Quantization

            # Environment configuration
            .with_env("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            .with_env("TF_GPU_MEMORY_ALLOCATION", "incremental")
        )

    @staticmethod
    def jax_image(version: str = "0.4.14"):
        """JAX optimized image."""
        return (
            Image(username="myuser", name="jax", tag=version)
            .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")
            .run_command("apt update && apt install -y python3 python3-pip")

            # Install JAX with CUDA
            .run_command(f"pip3 install jax[cuda11_local]=={version} -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
            .run_command("pip3 install flax optax")  # Common JAX libraries
            .run_command("pip3 install chex dm-haiku")  # DeepMind utilities

            # Performance libraries
            .run_command("pip3 install jaxlib")
            .run_command("pip3 install equinox")  # Neural networks in JAX
        )

# Usage examples
pytorch_img = AIFrameworkImages.pytorch_image("2.1.0", "11.8")
tf_img = AIFrameworkImages.tensorflow_image("2.13.0")
jax_img = AIFrameworkImages.jax_image("0.4.14")
```

## Performance Optimization

### Compilation and Caching

Optimize build times and runtime performance:

```python
def create_optimized_ai_image():
    """Create performance-optimized AI image."""

    return (
        Image(username="myuser", name="optimized-ai", tag="1.0")
        .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")

        # System optimizations
        .run_command("apt update && apt install -y python3 python3-pip build-essential")
        .run_command("apt install -y ccache")  # Compiler cache

        # Configure compilation cache
        .with_env("CCACHE_DIR", "/tmp/ccache")
        .with_env("CCACHE_MAXSIZE", "2G")

        # Python optimizations
        .with_env("PYTHONOPTIMIZE", "2")  # Enable optimizations
        .with_env("PYTHONDONTWRITEBYTECODE", "1")  # Don't write .pyc files

        # PyTorch compilation cache
        .with_env("TORCH_COMPILE_CACHE_DIR", "/tmp/torch_cache")
        .run_command("mkdir -p /tmp/torch_cache")

        # Install with optimizations
        .run_command("pip3 install --upgrade pip setuptools wheel")
        .run_command("CC='ccache gcc' pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

        # Compile frequently used kernels ahead of time
        .run_command("python3 -c 'import torch; torch.compile(torch.nn.Linear(10, 1))'")

        # Clean up build artifacts
        .run_command("apt remove -y build-essential && apt autoremove -y")
        .run_command("rm -rf /var/lib/apt/lists/* /tmp/ccache")
    )
```

### Memory Optimization

Create memory-efficient images:

```python
def create_memory_optimized_image():
    """Create memory-efficient image for resource-constrained environments."""

    return (
        Image(username="myuser", name="memory-optimized", tag="1.0")
        .from_base("python:3.11-slim")  # Smaller base image

        # Minimal system dependencies
        .run_command("apt update && apt install -y --no-install-recommends python3-dev gcc")

        # Install only essential packages
        .run_command("pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu")  # CPU-only for smaller size
        .run_command("pip3 install --no-cache-dir transformers[torch]")

        # Memory optimizations
        .with_env("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        .with_env("TRANSFORMERS_CACHE", "/tmp/transformers_cache")

        # Clean up
        .run_command("apt remove -y gcc python3-dev && apt autoremove -y")
        .run_command("rm -rf /var/lib/apt/lists/*")
        .run_command("pip3 cache purge")
    )
```

## Security Hardening

### Secure Base Images

Build security-hardened images:

```python
def create_secure_image():
    """Create security-hardened image."""

    return (
        Image(username="myuser", name="secure-ai", tag="1.0")
        .from_base("nvidia/cuda:11.8-runtime-ubuntu22.04")  # Runtime, not devel

        # Security updates
        .run_command("apt update && apt upgrade -y")

        # Install only necessary packages
        .run_command("apt install -y --no-install-recommends python3 python3-pip")

        # Create non-root user
        .run_command("groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser")
        .run_command("mkdir -p /app && chown appuser:appgroup /app")

        # Remove unnecessary packages and files
        .run_command("apt remove -y --purge wget curl && apt autoremove -y")
        .run_command("rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*")

        # Security configurations
        .run_command("chmod 755 /app")
        .run_command("find /usr -type f -perm +6000 -exec chmod -s {} \\; || true")  # Remove setuid/setgid

        # Switch to non-root user
        .with_user("appuser")
        .with_workdir("/app")

        # Security environment variables
        .with_env("PYTHONDONTWRITEBYTECODE", "1")
        .with_env("PYTHONUNBUFFERED", "1")
    )
```

### Secrets Management

Handle secrets securely in images:

```python
def create_image_with_secrets():
    """Create image with proper secrets handling."""

    return (
        Image(username="myuser", name="secure-secrets", tag="1.0")
        .from_base("python:3.11-slim")

        # Install secrets management tools
        .run_command("pip3 install cryptography python-dotenv")

        # Create secrets directory with proper permissions
        .run_command("mkdir -p /app/secrets && chmod 700 /app/secrets")

        # Never embed secrets in image layers!
        # Use environment variables or mounted volumes instead
        .with_env("SECRETS_PATH", "/app/secrets")

        # Configure for external secret injection
        .run_command("echo '#!/bin/bash\n"
                    "if [ -f /app/secrets/.env ]; then\n"
                    "  export $(cat /app/secrets/.env | grep -v ^# | xargs)\n"
                    "fi\n"
                    "exec \"$@\"' > /app/entrypoint.sh")
        .run_command("chmod +x /app/entrypoint.sh")

        # Use entrypoint for secret loading
        .with_entrypoint(["/app/entrypoint.sh"])
    )
```

## Specialized Image Types

### Development Images

Create development-friendly images with debugging tools:

```python
def create_development_image():
    """Create development image with debugging tools."""

    return (
        Image(username="myuser", name="dev-ai", tag="latest")
        .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")

        # Development tools
        .run_command("apt update && apt install -y python3 python3-pip git vim curl htop")
        .run_command("apt install -y iputils-ping net-tools strace gdb")

        # Python development tools
        .run_command("pip3 install ipython jupyter notebook")
        .run_command("pip3 install debugpy pytest pytest-cov")
        .run_command("pip3 install black isort flake8 mypy")

        # AI libraries with debug symbols
        .run_command("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        .run_command("pip3 install transformers[dev]")

        # Jupyter configuration
        .run_command("jupyter notebook --generate-config")
        .run_command("echo \"c.NotebookApp.ip = '0.0.0.0'\" >> ~/.jupyter/jupyter_notebook_config.py")
        .run_command("echo \"c.NotebookApp.token = ''\" >> ~/.jupyter/jupyter_notebook_config.py")

        # Development environment
        .with_env("PYTHONPATH", "/app")
        .with_env("JUPYTER_ENABLE_LAB", "yes")
        .with_workdir("/app")

        # Expose Jupyter port
        .expose_port(8888)
    )
```

### Production Images

Create production-optimized images:

```python
def create_production_image():
    """Create production-ready image."""

    return (
        Image(username="myuser", name="prod-ai", tag="1.0")
        .from_base("nvidia/cuda:11.8-runtime-ubuntu22.04")  # Runtime only

        # Minimal production dependencies
        .run_command("apt update && apt install -y --no-install-recommends python3 python3-pip")

        # Production Python packages
        .run_command("pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        .run_command("pip3 install --no-cache-dir transformers accelerate")
        .run_command("pip3 install --no-cache-dir fastapi uvicorn[standard]")

        # Production optimizations
        .with_env("PYTHONOPTIMIZE", "2")
        .with_env("PYTHONDONTWRITEBYTECODE", "1")
        .with_env("PYTHONUNBUFFERED", "1")

        # Health check script
        .run_command("echo '#!/bin/bash\ncurl -f http://localhost:8000/health || exit 1' > /app/healthcheck.sh")
        .run_command("chmod +x /app/healthcheck.sh")

        # Non-root user for security
        .run_command("useradd -m -u 1000 appuser")
        .run_command("mkdir -p /app && chown appuser:appuser /app")
        .with_user("appuser")

        # Clean up
        .run_command("rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*")

        # Health check
        .with_healthcheck(["CMD", "/app/healthcheck.sh"])
    )
```

## Image Management and Versioning

### Semantic Versioning

Implement proper versioning for images:

```python
class VersionedImageBuilder:
    """Build images with semantic versioning."""

    def __init__(self, username: str, name: str):
        self.username = username
        self.name = name
        self.major = 1
        self.minor = 0
        self.patch = 0
        self.build = None

    def version(self, major: int, minor: int, patch: int, build: str = None):
        """Set version numbers."""
        self.major = major
        self.minor = minor
        self.patch = patch
        self.build = build
        return self

    def get_version_tag(self) -> str:
        """Get formatted version tag."""
        tag = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            tag += f"-{self.build}"
        return tag

    def build_image(self, base_config_func):
        """Build image with version tags."""
        version_tag = self.get_version_tag()

        image = base_config_func(
            Image(self.username, self.name, version_tag)
        )

        # Add version metadata
        image = (
            image
            .with_label("version", version_tag)
            .with_label("major", str(self.major))
            .with_label("minor", str(self.minor))
            .with_label("patch", str(self.patch))
        )

        if self.build:
            image = image.with_label("build", self.build)

        return image

# Usage
def my_ai_config(image: Image) -> Image:
    return (
        image
        .from_base("nvidia/cuda:11.8-runtime-ubuntu22.04")
        .run_command("pip3 install torch transformers")
    )

builder = VersionedImageBuilder("myuser", "my-ai-app")
image = builder.version(2, 1, 0, "beta").build_image(my_ai_config)
```

### Environment-Specific Images

Build images for different environments:

```python
class EnvironmentImageBuilder:
    """Build environment-specific images."""

    @staticmethod
    def development(base_image: Image) -> Image:
        """Development environment configuration."""
        return (
            base_image
            .run_command("pip3 install ipython jupyter pytest debugpy")
            .with_env("FLASK_ENV", "development")
            .with_env("LOG_LEVEL", "DEBUG")
            .expose_port(8888)  # Jupyter
            .expose_port(5678)  # Debugger
        )

    @staticmethod
    def staging(base_image: Image) -> Image:
        """Staging environment configuration."""
        return (
            base_image
            .with_env("FLASK_ENV", "staging")
            .with_env("LOG_LEVEL", "INFO")
            .with_healthcheck(["CMD", "curl", "-f", "http://localhost:8000/health"])
        )

    @staticmethod
    def production(base_image: Image) -> Image:
        """Production environment configuration."""
        return (
            base_image
            .with_env("FLASK_ENV", "production")
            .with_env("LOG_LEVEL", "WARNING")
            .with_env("PYTHONOPTIMIZE", "2")
            .run_command("pip3 cache purge")  # Clean up cache
            .with_healthcheck(["CMD", "curl", "-f", "http://localhost:8000/health"])
        )

# Usage
base = Image("myuser", "my-app", "1.0").from_base("python:3.11-slim")

dev_image = EnvironmentImageBuilder.development(base)
staging_image = EnvironmentImageBuilder.staging(base)
prod_image = EnvironmentImageBuilder.production(base)
```

## Testing and Validation

### Image Testing Framework

Test images before deployment:

```python
import subprocess
import tempfile
import json

class ImageTester:
    """Test framework for validating images."""

    def __init__(self, image: Image):
        self.image = image
        self.test_results = []

    def test_python_imports(self, packages: list):
        """Test that Python packages can be imported."""
        test_script = f"""
import sys
failed_imports = []
for package in {packages}:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        failed_imports.append((package, str(e)))
        print(f"✗ {package}: {e}")

if failed_imports:
    sys.exit(1)
"""

        result = self._run_test_script(test_script)
        self.test_results.append({
            "test": "python_imports",
            "passed": result.returncode == 0,
            "output": result.stdout
        })
        return result.returncode == 0

    def test_gpu_availability(self):
        """Test GPU availability and CUDA setup."""
        test_script = """
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
else:
    print("CUDA not available")
    sys.exit(1)
"""

        result = self._run_test_script(test_script)
        self.test_results.append({
            "test": "gpu_availability",
            "passed": result.returncode == 0,
            "output": result.stdout
        })
        return result.returncode == 0

    def test_model_loading(self, model_name: str):
        """Test that a specific model can be loaded."""
        test_script = f"""
from transformers import AutoTokenizer, AutoModel
import sys

try:
    tokenizer = AutoTokenizer.from_pretrained("{model_name}")
    model = AutoModel.from_pretrained("{model_name}")
    print(f"✓ Successfully loaded {model_name}")
    print(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")
except Exception as e:
    print(f"✗ Failed to load {model_name}: {{e}}")
    sys.exit(1)
"""

        result = self._run_test_script(test_script)
        self.test_results.append({
            "test": f"model_loading_{model_name}",
            "passed": result.returncode == 0,
            "output": result.stdout
        })
        return result.returncode == 0

    def test_security(self):
        """Test security configurations."""
        test_script = """
import os
import pwd
import sys

# Check user
user = pwd.getpwuid(os.getuid())
print(f"Running as user: {user.pw_name} (UID: {user.pw_uid})")

if user.pw_uid == 0:
    print("✗ Running as root - security risk!")
    sys.exit(1)
else:
    print("✓ Running as non-root user")

# Check write permissions
write_paths = ["/", "/etc", "/usr"]
for path in write_paths:
    if os.access(path, os.W_OK):
        print(f"✗ Write access to {path} - security risk!")
        sys.exit(1)
    else:
        print(f"✓ No write access to {path}")
"""

        result = self._run_test_script(test_script)
        self.test_results.append({
            "test": "security",
            "passed": result.returncode == 0,
            "output": result.stdout
        })
        return result.returncode == 0

    def _run_test_script(self, script: str):
        """Run a test script in a container."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            # This would run the script in the container
            # For actual implementation, you'd use docker run or similar
            result = subprocess.run([
                "python3", script_path
            ], capture_output=True, text=True, timeout=60)

            return result
        finally:
            os.unlink(script_path)

    def run_all_tests(self):
        """Run all tests and return summary."""
        tests_passed = 0
        total_tests = len(self.test_results)

        for result in self.test_results:
            if result["passed"]:
                tests_passed += 1

        return {
            "total_tests": total_tests,
            "tests_passed": tests_passed,
            "success_rate": tests_passed / total_tests if total_tests > 0 else 0,
            "results": self.test_results
        }

# Usage
image = Image("myuser", "test-image", "1.0")
tester = ImageTester(image)

tester.test_python_imports(["torch", "transformers", "numpy"])
tester.test_gpu_availability()
tester.test_model_loading("bert-base-uncased")
tester.test_security()

summary = tester.run_all_tests()
print(f"Tests passed: {summary['tests_passed']}/{summary['total_tests']}")
```

## Troubleshooting and Debugging

### Common Build Issues

Debug common image building problems:

```python
class ImageDebugger:
    """Debug common image building issues."""

    @staticmethod
    def diagnose_build_failure(build_log: str):
        """Analyze build log for common issues."""

        issues = []

        # Check for common problems
        if "E: Package" in build_log and "has no installation candidate" in build_log:
            issues.append({
                "issue": "Package not found",
                "solution": "Update package lists with 'apt update' before installing packages"
            })

        if "Permission denied" in build_log:
            issues.append({
                "issue": "Permission denied",
                "solution": "Ensure user has proper permissions or run as root for system operations"
            })

        if "No space left on device" in build_log:
            issues.append({
                "issue": "Disk space",
                "solution": "Clean up unused files and caches, or increase disk space"
            })

        if "CUDA_ERROR_OUT_OF_MEMORY" in build_log:
            issues.append({
                "issue": "GPU memory insufficient",
                "solution": "Reduce batch size or use a GPU with more memory"
            })

        if "ModuleNotFoundError" in build_log:
            issues.append({
                "issue": "Python module not found",
                "solution": "Install missing dependencies or check PYTHONPATH"
            })

        return issues

    @staticmethod
    def suggest_optimizations(image_size_mb: int, build_time_seconds: int):
        """Suggest optimizations based on image metrics."""

        suggestions = []

        if image_size_mb > 5000:  # > 5GB
            suggestions.append("Image is large - consider multi-stage builds or smaller base images")

        if build_time_seconds > 600:  # > 10 minutes
            suggestions.append("Build is slow - consider using pre-built base images or build caching")

        suggestions.extend([
            "Combine RUN commands to reduce layers",
            "Clean up package caches and temporary files",
            "Use .dockerignore to exclude unnecessary files",
            "Order commands from least to most likely to change"
        ])

        return suggestions

# Usage
debugger = ImageDebugger()
issues = debugger.diagnose_build_failure(build_log_content)
suggestions = debugger.suggest_optimizations(8000, 800)
```

### Build Optimization

Optimize build performance:

```python
def create_optimized_build_image():
    """Create image with build optimizations."""

    return (
        Image(username="myuser", name="optimized-build", tag="1.0")
        .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")

        # Layer optimization - combine related commands
        .run_command(
            "apt update && "
            "apt install -y python3 python3-pip git && "
            "rm -rf /var/lib/apt/lists/*"  # Clean up in same layer
        )

        # Use build cache effectively
        .copy_file("requirements.txt", "/tmp/requirements.txt")  # Copy requirements first
        .run_command("pip3 install -r /tmp/requirements.txt")    # Install deps
        .copy_file(".", "/app")                                  # Copy code last

        # Build-time variables for optimization
        .with_arg("MAKEFLAGS", "-j$(nproc)")  # Parallel compilation
        .with_arg("PIP_NO_CACHE_DIR", "1")    # Don't cache pip downloads

        # Multi-stage friendly structure
        .with_label("stage", "build")
        .with_workdir("/app")
    )
```

## Best Practices Summary

### Image Building Checklist

```python
class ImageBuildingChecklist:
    """Comprehensive checklist for image building best practices."""

    def __init__(self):
        self.checks = {
            "security": [
                "Use non-root user",
                "Remove setuid/setgid binaries",
                "Don't embed secrets",
                "Use minimal base images",
                "Keep system packages updated"
            ],

            "performance": [
                "Use appropriate base image",
                "Minimize layers",
                "Leverage build cache",
                "Clean up in same layer",
                "Use multi-stage builds"
            ],

            "maintainability": [
                "Pin package versions",
                "Use semantic versioning",
                "Add descriptive labels",
                "Document custom configurations",
                "Include health checks"
            ],

            "size_optimization": [
                "Remove package caches",
                "Use slim base images",
                "Avoid unnecessary dependencies",
                "Compress layers where possible",
                "Use .dockerignore"
            ]
        }

    def validate_image(self, image: Image) -> dict:
        """Validate image against best practices."""

        # This would inspect the image and check against the checklist
        # For demo purposes, returning a structure

        return {
            "security_score": 85,
            "performance_score": 90,
            "maintainability_score": 80,
            "size_score": 75,
            "recommendations": [
                "Consider using non-root user",
                "Add health check",
                "Clean up package caches"
            ]
        }

# Usage
checklist = ImageBuildingChecklist()
image = Image("myuser", "my-app", "1.0")
validation = checklist.validate_image(image)
print(f"Overall score: {sum(validation.values()[:4]) / 4}")
```

## Next Steps

- **Advanced Patterns**: Explore multi-stage builds and image optimization
- **CI/CD Integration**: Automate image building and testing
- **Registry Management**: Manage image repositories and distributions
- **Security Scanning**: Implement vulnerability scanning in build pipeline

For more advanced topics, see:

- [Custom Chutes Guide](custom-chutes)
- [Best Practices](best-practices)
- [Security Guide](security)
