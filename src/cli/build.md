# Building Images

The `chutes build` command creates Docker images for your chutes with all necessary dependencies and optimizations for the Chutes platform.

## Basic Build Command

### `chutes build`

Build a Docker image for your chute.

```bash
chutes build <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image
- `--local`: Build locally instead of remotely
- `--debug`: Enable debug logging
- `--include-cwd`: Include entire current directory
- `--wait`: Wait for build to complete
- `--public`: Mark image as public
- `--tag TEXT`: Custom tag for the image
- `--no-cache`: Disable build cache

## Build Examples

### Basic Remote Build

```bash
# Build on Chutes infrastructure (recommended)
chutes build my_chute:chute --wait
```

**Benefits of Remote Building:**

- 🚀 Faster build times with powerful infrastructure
- 📦 Optimized caching and layer sharing
- 🔒 Secure build environment
- 💰 No local resource usage

### Local Development Build

```bash
# Build locally for testing and development
chutes build my_chute:chute --local --debug
```

**When to Use Local Builds:**

- 🧪 Quick development iterations
- 🔍 Debugging build issues
- 🌐 Limited internet connectivity
- 🔒 Sensitive code that shouldn't leave your machine

### Production Build with Assets

```bash
# Build with logo and make public
chutes build my_chute:chute --logo ./assets/logo.png --public --wait
```

### Force Clean Build

```bash
# Build without cache for clean rebuild
chutes build my_chute:chute --no-cache --wait
```

## Build Process

### What Happens During Build

1. **Code Analysis**: Chutes analyzes your Python code and dependencies
2. **Image Creation**: Generates optimized Dockerfile
3. **Dependency Installation**: Installs Python packages and system dependencies
4. **Model Downloads**: Pre-downloads AI models if specified
5. **Optimization**: Applies platform-specific optimizations
6. **Validation**: Tests the built image for compatibility

### Build Stages

```bash
# Example build output
Building chute: my_chute:chute
✓ Analyzing code structure
✓ Creating base image
✓ Installing system dependencies
✓ Setting up Python environment
✓ Installing Python packages
✓ Downloading models
✓ Applying optimizations
✓ Running validation tests
✓ Pushing to registry

Build completed successfully!
Image: chutes.ai/myuser/my_chute:latest
```

## Advanced Build Options

### Custom Tags

```bash
# Build with custom version tag
chutes build my_chute:chute --tag v1.2.0

# Build with multiple tags
chutes build my_chute:chute --tag latest --tag stable
```

### Including Files

```bash
# Include entire current directory
chutes build my_chute:chute --include-cwd

# Include specific files (use .chutesbuildignore)
echo "*.pyc" > .chutesbuildignore
echo "__pycache__/" >> .chutesbuildignore
echo ".git/" >> .chutesbuildignore
```

### Debug Builds

```bash
# Enable verbose logging
chutes build my_chute:chute --debug --local

# Keep intermediate containers for inspection
chutes build my_chute:chute --debug --keep-intermediate
```

## Build Configuration

### .chutesbuildignore

Create a `.chutesbuildignore` file to exclude files from builds:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large files
*.mp4
*.avi
dataset/
models/

# Secrets
.env
*.key
config.ini
```

### Build Context Optimization

```python
# In your chute file, optimize image building
from chutes.image import Image

# Efficient layering for faster builds
image = (
    Image(username="myuser", name="my-chute", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")

    # Install system deps first (rarely change)
    .run_command("apt-get update && apt-get install -y git curl")

    # Install Python (stable)
    .with_python("3.11")

    # Install core ML packages (change less frequently)
    .run_command("pip install torch==2.1.0 transformers==4.30.0")

    # Install app-specific packages (copy requirements first)
    .add("requirements.txt", "/app/requirements.txt")
    .run_command("pip install -r /app/requirements.txt")

    # Copy application code last (changes most frequently)
    .add("src/", "/app/src/")
)
```

## Performance Optimization

### Build Caching

```bash
# Use build cache for faster builds (default)
chutes build my_chute:chute --wait

# Clear cache for clean build
chutes build my_chute:chute --no-cache

# Use local cache
chutes build my_chute:chute --local --cache-from my-cache-image
```

### Parallel Builds

```bash
# Build multiple chutes in parallel
chutes build app1:chute --wait &
chutes build app2:chute --wait &
chutes build app3:chute --wait &
wait

echo "All builds completed"
```

### Resource Limits

```bash
# Local builds with resource limits
docker system prune -f  # Clean up before building
chutes build my_chute:chute --local --memory 8g --cpus 4
```

## Troubleshooting Builds

### Common Build Issues

**Build fails with dependency errors?**

```bash
# Check requirements.txt
cat requirements.txt

# Build with debug to see full output
chutes build my_chute:chute --local --debug

# Try building with no cache
chutes build my_chute:chute --no-cache --debug
```

**Out of memory during build?**

```bash
# For local builds, check available memory
free -h

# Use remote building for large models
chutes build my_chute:chute --wait  # Remote has more memory

# Optimize image layers
# Put large downloads in separate layers
```

**Build takes too long?**

```bash
# Use remote building (usually faster)
chutes build my_chute:chute --wait

# Optimize Docker layers in your Image definition
# Check .chutesbuildignore to exclude unnecessary files

# Use smaller base images where possible
```

**Permission errors?**

```bash
# Check file permissions
ls -la

# Fix permissions if needed
chmod -R 755 .

# For local builds, check Docker daemon
sudo systemctl status docker
```

### Debug Commands

```bash
# Inspect build context
tar -czf - . | tar -tz | head -20

# Check image layers
docker history myuser/my-chute:latest

# Inspect built image
docker run -it myuser/my-chute:latest /bin/bash

# Check build logs
chutes build my_chute:chute --debug 2>&1 | tee build.log
```

## Build Strategies

### Development Workflow

```bash
# Fast iteration during development
chutes build my_chute:chute --local --tag dev

# Test the built image
chutes run my_chute:chute --tag dev

# Once stable, build remotely
chutes build my_chute:chute --wait --tag stable
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Chutes
        run: pip install chutes

      - name: Configure Chutes
        env:
          CHUTES_API_KEY: ${{ secrets.CHUTES_API_KEY }}
        run: |
          mkdir -p ~/.chutes
          echo "[auth]" > ~/.chutes/config.ini
          echo "api_key = $CHUTES_API_KEY" >> ~/.chutes/config.ini

      - name: Build Image
        run: |
          chutes build my_app:chute --wait --tag ${{ github.sha }}
          chutes build my_app:chute --wait --tag latest
```

### Production Builds

```bash
# Production build checklist
echo "Building production image..."

# 1. Clean workspace
git status --porcelain
[ $? -eq 0 ] || { echo "Uncommitted changes found"; exit 1; }

# 2. Run tests
python -m pytest tests/

# 3. Build with version tag
VERSION=$(git describe --tags --abbrev=0)
chutes build my_chute:chute --tag $VERSION --wait

# 4. Tag as latest if on main branch
if [ "$(git branch --show-current)" = "main" ]; then
    chutes build my_chute:chute --tag latest --wait
fi

echo "Production build completed: $VERSION"
```

## Multi-Stage Builds

For complex applications, use multi-stage builds:

```python
# Build stage
build_image = (
    Image(username="myuser", name="builder", tag="temp")
    .from_base("nvidia/cuda:12.2-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install build-tools")
    .add("src/", "/build/src/")
    .run_command("cd /build && python setup.py build_ext")
)

# Production stage
production_image = (
    Image(username="myuser", name="my-app", tag="1.0")
    .from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")
    .with_python("3.11")
    .copy_from_image(build_image, "/build/dist/", "/app/")
    .run_command("pip install "runtime-deps")
)
```

## Build Monitoring

### Build Metrics

```bash
# Monitor build progress
chutes build my_chute:chute --wait --progress

# Check build history
chutes builds list --filter my_chute

# Get build details
chutes builds get <build_id>
```

### Build Notifications

```bash
# Build with notifications
chutes build my_chute:chute --wait --notify-email
chutes build my_chute:chute --wait --notify-slack webhook_url
```

## Best Practices

### 1. **Optimize Layer Caching**

```python
# Good: Stable operations first
.with_python("3.11")
.with_pip_packages("torch==2.1.0")  # Pin versions
.add("requirements.txt", "/app/")
.run_command("pip install -r /app/requirements.txt")
.add("src/", "/app/src/")  # Code changes most

# Bad: Frequent changes first
.add("src/", "/app/src/")  # This invalidates all subsequent layers
.with_pip_packages("torch")
```

### 2. **Pin Dependencies**

```txt
# requirements.txt - Good
torch==2.1.0
transformers==4.30.2
numpy==1.24.3

# requirements.txt - Bad
torch
transformers
numpy  # Could break with version changes
```

### 3. **Minimize Image Size**

```python
# Use multi-stage builds for smaller images
# Clean up package caches
.run_command("""
    apt-get update &&
    apt-get install -y git curl &&
    rm -rf /var/lib/apt/lists/*
""")

# Use .chutesbuildignore extensively
```

### 4. **Security Scanning**

```bash
# Scan images for vulnerabilities
chutes build my_chute:chute --scan-security

# Use official base images
.from_base("nvidia/cuda:12.2-runtime-ubuntu22.04")  # Official
```

## Next Steps

- **[Deploying Chutes](/docs/cli/deploy)** - Deploy your built images
- **[Managing Resources](/docs/cli/manage)** - Manage your chutes
- **[Account Management](/docs/cli/account)** - API keys and configuration
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
