# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when developing and deploying with Chutes.

## Deployment Issues

### Build Failures

#### Python Package Installation Errors

**Problem**: Packages fail to install during image build

```bash
ERROR: Could not find a version that satisfies the requirement torch==2.1.0
```

**Solutions**:

```python
from chutes.image import Image

# Use compatible base images
image = Image(
    base_image="nvidia/cuda:12.1-devel-ubuntu22.04",
    python_version="3.11"
)

# Specify compatible package versions
image.pip_install([
    "torch==2.1.0",
    "torchvision==0.16.0",
    "--extra-index-url https://download.pytorch.org/whl/cu121"
])

# Alternative: Use conda for complex dependencies
image.run_command("conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia")
```

#### Docker Build Context Issues

**Problem**: Large files causing slow uploads

```bash
Uploading build context... 2.3GB
```

**Solutions**:

```python
# Create .dockerignore to exclude unnecessary files
# .dockerignore content:
"""
__pycache__/
*.pyc
.git/
.pytest_cache/
large_datasets/
*.mp4
*.avi
"""

# Or use specific file inclusion
image.add("app.py", "/app/app.py")
image.add("requirements.txt", "/app/requirements.txt")
```

#### Permission Errors

**Problem**: Permission denied during build

```bash
Permission denied: '/usr/local/bin/pip'
```

**Solutions**:

```python
# Run commands as root when needed
image.run_command("apt-get update && apt-get install -y curl", user="root")

# Set proper ownership
image.run_command("chown -R chutes:chutes /app", user="root")

# Use USER directive correctly
image.user("chutes")
```

### Deployment Timeouts

**Problem**: Deployment hangs or times out

**Solutions**:

```python
# Optimize startup time
@chute.on_startup()
async def setup(self):
    # Move heavy operations to background
    asyncio.create_task(self.load_model_async())

async def load_model_async(self):
    """Load model in background to avoid startup timeout."""
    self.model = load_large_model()
    self.ready = True

@chute.cord(public_api_path="/health")
async def health_check(self):
    """Health check endpoint."""
    return {"status": "ready" if hasattr(self, 'ready') else "loading"}
```

## Runtime Errors

### Out of Memory Errors

#### GPU Out of Memory

**Problem**: CUDA out of memory errors

```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

```python
import torch
import gc

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce batch size
@chute.cord(public_api_path="/generate")
async def generate(self, request: GenerateRequest):
    # Process in smaller batches
    batch_size = min(request.batch_size, 4)

    # Use mixed precision
    with torch.cuda.amp.autocast():
        outputs = model.generate(**inputs)

    return outputs

# Optimize node selector
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,  # Increase VRAM requirement
    include=["a100", "h100"]
)
```

#### System RAM Issues

**Problem**: System runs out of RAM

```python
MemoryError: Unable to allocate array
```

**Solutions**:

```python
# Increase RAM in node selector
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24
)

# Use memory-efficient data loading
import torch.utils.data as data

class MemoryEfficientDataset(data.Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __getitem__(self, idx):
        # Load data on-demand instead of pre-loading
        return load_data(self.file_paths[idx])
```

### Model Loading Errors

#### Missing Model Files

**Problem**: Model files not found

```python
FileNotFoundError: Model file not found: /models/pytorch_model.bin
```

**Solutions**:

```python
from huggingface_hub import snapshot_download
import os

@chute.on_startup()
async def setup(self):
    """Download model if not present."""
    model_path = "/models/my-model"

    if not os.path.exists(model_path):
        # Download model during startup
        snapshot_download(
            repo_id="microsoft/DialoGPT-medium",
            local_dir=model_path,
            token=os.getenv("HF_TOKEN")  # If private model
        )

    self.model = load_model(model_path)
```

#### Model Compatibility Issues

**Problem**: Model format incompatible with library version

```python
ValueError: Unsupported model format
```

**Solutions**:

```python
# Pin compatible versions
image.pip_install([
    "transformers==4.36.0",
    "torch==2.1.0",
    "safetensors==0.4.0"
])

# Use format conversion
from transformers import AutoModel
import torch

# Convert to compatible format
model = AutoModel.from_pretrained("model-name")
torch.save(model.state_dict(), "/models/converted_model.pt")
```

## Performance Problems

### Slow Inference

**Problem**: Inference takes too long

**Diagnosis**:

```python
import time
import torch

@chute.cord(public_api_path="/generate")
async def generate(self, request: GenerateRequest):
    start_time = time.time()

    # Profile different stages
    load_time = time.time()
    inputs = prepare_inputs(request.text)
    prep_time = time.time() - load_time

    # Inference timing
    inference_start = time.time()
    with torch.no_grad():
        outputs = self.model.generate(**inputs)
    inference_time = time.time() - inference_start

    # Post-processing timing
    post_start = time.time()
    result = postprocess_outputs(outputs)
    post_time = time.time() - post_start

    total_time = time.time() - start_time

    self.logger.info(f"Timing - Prep: {prep_time:.2f}s, Inference: {inference_time:.2f}s, Post: {post_time:.2f}s, Total: {total_time:.2f}s")

    return result
```

**Solutions**:

```python
# Enable optimizations
model.eval()
model = torch.compile(model)  # PyTorch 2.0+ optimization

# Use efficient data types
model = model.half()  # Use FP16

# Batch processing
@chute.cord(public_api_path="/batch_generate")
async def batch_generate(self, requests: List[GenerateRequest]):
    # Process multiple requests together
    batch_inputs = [prepare_inputs(req.text) for req in requests]
    batch_outputs = self.model.generate_batch(batch_inputs)
    return [postprocess_outputs(output) for output in batch_outputs]
```

### High Latency

**Problem**: First request is very slow (cold start)

**Solutions**:

```python
@chute.on_startup()
async def setup(self):
    """Warm up model to reduce cold start."""
    self.model = load_model()

    # Warm-up inference
    dummy_input = "Hello world"
    _ = self.model.generate(dummy_input)

    self.logger.info("Model warmed up successfully")

# Use model caching
@chute.cord(public_api_path="/generate")
async def generate(self, request: GenerateRequest):
    # Cache compiled model
    if not hasattr(self, '_compiled_model'):
        self._compiled_model = torch.compile(self.model)

    return self._compiled_model.generate(request.text)
```

## Authentication Issues

### API Key Problems

**Problem**: Authentication failures

```bash
HTTPException: 401 Unauthorized
```

**Solutions**:

```bash
# Check API key configuration
chutes auth status

# Set API key correctly
chutes auth login
# or
export CHUTES_API_KEY="your-api-key"

# Verify key is working
chutes chutes list
```

### Permission Errors

**Problem**: Insufficient permissions for operations

```bash
HTTPException: 403 Forbidden
```

**Solutions**:

```bash
# Check account permissions
chutes account info

# Contact support if you need additional permissions
# Ensure you're using the correct username in deployments
```

## Debugging Techniques

### Logging and Monitoring

```python
import logging
from chutes.chute import Chute

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)

chute = Chute(
    username="myuser",
    name="debug-app"
)

@chute.on_startup()
async def setup(self):
    self.logger.info("Application starting up")

    # Log system information
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            self.logger.info(f"GPU {i}: {props.name} ({props.total_memory // (1024**3)}GB)")

@chute.cord(public_api_path="/debug")
async def debug_info(self):
    """Debug endpoint for system information."""
    import psutil
    import torch

    info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_memory": {}
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            info["gpu_memory"][f"gpu_{i}"] = {
                "allocated_gb": allocated / (1024**3),
                "total_gb": total / (1024**3),
                "utilization": (allocated / total) * 100
            }

    return info
```

### Remote Debugging

```python
# Enable remote debugging for development
import os

if os.getenv("DEBUG_MODE"):
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
```

### Error Tracking

```python
import traceback
from chutes.exception import ChuteException

@chute.cord(public_api_path="/generate")
async def generate(self, request: GenerateRequest):
    try:
        result = self.model.generate(request.text)
        return result
    except torch.cuda.OutOfMemoryError:
        self.logger.error("GPU out of memory", exc_info=True)
        raise ChuteException(
            status_code=503,
            detail="Service temporarily unavailable due to memory constraints"
        )
    except Exception as e:
        self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise ChuteException(
            status_code=500,
            detail="Internal server error"
        )
```

## Resource Issues

### Node Selection Problems

**Problem**: No available nodes matching requirements

**Solutions**:

```python
# Make node selector more flexible
node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,  # Reduce if too restrictive
    # Don't restrict VRAM to allow larger GPUs
    include=["a100", "l40", "a6000"],  # Include more GPU types
    exclude=[]  # Remove exclusions
)
```

### Scaling Issues

**Problem**: Chute can't handle high load

**Solutions**:

```python
# Optimize for concurrency
node_selector = NodeSelector(
    gpu_count=2,  # Multiple GPUs for parallel processing
    min_vram_gb_per_gpu=24
)

# Implement request queuing
import asyncio
from asyncio import Semaphore

class RateLimitedChute(Chute):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = Semaphore(5)  # Limit concurrent requests

    @chute.cord(public_api_path="/generate")
    async def generate(self, request: GenerateRequest):
        async with self.semaphore:
            return await self._generate_impl(request)
```

## Networking Problems

### Connection Issues

**Problem**: Cannot reach deployed chute

**Solutions**:

```bash
# Check chute status
chutes chutes get myuser/my-chute

# Check logs for errors
chutes chutes logs myuser/my-chute

# Test health endpoint
curl https://your-chute-url/health
```

### Timeout Issues

**Problem**: Requests timing out

**Solutions**:

```python
# Implement async processing for long-running tasks
@chute.job()
async def process_long_task(self, task_id: str, input_data: dict):
    """Background job for long-running tasks."""
    try:
        result = await long_running_process(input_data)
        # Store result in database or file system
        store_result(task_id, result)
    except Exception as e:
        self.logger.error(f"Task {task_id} failed: {e}")
        store_error(task_id, str(e))

@chute.cord(public_api_path="/start_task")
async def start_task(self, request: TaskRequest):
    """Start a background task and return task ID."""
    task_id = generate_task_id()
    await self.process_long_task(task_id, request.data)
    return {"task_id": task_id, "status": "started"}

@chute.cord(public_api_path="/task_status/{task_id}")
async def get_task_status(self, task_id: str):
    """Get status of a background task."""
    return get_task_status(task_id)
```

## Model Loading Issues

### Download Failures

**Problem**: Model download fails during startup

**Solutions**:

```python
import os
import time
from huggingface_hub import snapshot_download

@chute.on_startup()
async def setup(self):
    """Robust model loading with retries."""
    model_path = "/models/my-model"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if not os.path.exists(model_path):
                self.logger.info(f"Downloading model (attempt {attempt + 1}/{max_retries})")
                snapshot_download(
                    repo_id="microsoft/DialoGPT-medium",
                    local_dir=model_path,
                    resume_download=True,  # Resume partial downloads
                    timeout=300  # 5 minute timeout
                )

            self.logger.info("Loading model...")
            self.model = load_model(model_path)
            self.logger.info("Model loaded successfully")
            break

        except Exception as e:
            self.logger.error(f"Model loading attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(30)  # Wait before retry
```

### Version Conflicts

**Problem**: Model requires different library versions

**Solutions**:

```python
# Create version-specific environments
image.run_command("pip install transformers==4.21.0 --target /app/transformers_old")

# Use specific versions dynamically
import sys
import os

@chute.on_startup()
async def setup(self):
    # Add specific library version to path
    sys.path.insert(0, "/app/transformers_old")

    # Now import the specific version
    import transformers
    self.logger.info(f"Using transformers version: {transformers.__version__}")
```

## Common Error Messages and Solutions

### "Module not found" Errors

```bash
ModuleNotFoundError: No module named 'transformers'
```

**Solution**: Add missing packages to image

```python
image.pip_install(["transformers", "torch", "tokenizers"])
```

### "CUDA device-side assert triggered"

**Solution**: Check tensor dimensions and data types

```python
# Ensure tensors are on correct device
inputs = {k: v.to(self.device) for k, v in inputs.items()}

# Check for invalid indices
assert torch.all(input_ids >= 0) and torch.all(input_ids < vocab_size)
```

### "HTTP 422 Unprocessable Entity"

**Solution**: Validate input schema

```python
from pydantic import BaseModel, validator

class GenerateRequest(BaseModel):
    text: str
    max_length: int = 100

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v

    @validator('max_length')
    def max_length_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Max length must be positive')
        return v
```

## Getting Help

### Log Analysis

```bash
# Get recent logs
chutes chutes logs myuser/my-chute --tail 100

# Follow logs in real-time
chutes chutes logs myuser/my-chute --follow

# Filter logs by level
chutes chutes logs myuser/my-chute | grep ERROR
```

### Diagnostic Commands

```bash
# Check account status
chutes auth status

# List deployed chutes
chutes chutes list

# Get detailed chute information
chutes chutes get myuser/my-chute

# Check system resources
chutes chutes metrics myuser/my-chute
```

### Support Resources

1. **Documentation**: Review relevant guides and API references
2. **Community**: Join the community forum for peer support
3. **Support Tickets**: Submit detailed bug reports with:
   - Full error messages
   - Relevant code snippets
   - Log excerpts
   - System information

### Reporting Issues

When reporting issues, include:

```python
# System information
import torch
import sys
import platform

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
print(f"Platform: {platform.platform()}")
```

This troubleshooting guide should help you resolve most common issues. For persistent problems, don't hesitate to reach out to support with detailed information about your setup and the specific error you're encountering.
