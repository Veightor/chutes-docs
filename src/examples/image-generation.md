# Image Generation with Diffusion Models

This guide demonstrates how to build powerful image generation services using state-of-the-art diffusion models like FLUX.1. You'll learn to create a complete image generation API with custom parameters, validation, and optimization.

## Overview

The Chutes platform makes it easy to deploy advanced image generation models:

- **FLUX.1 [dev]**: 12 billion parameter rectified flow transformer
- **Stable Diffusion**: Various versions and fine-tuned models
- **Custom Models**: Support for any diffusion architecture
- **GPU Optimization**: Automatic scaling and memory management

## Complete FLUX.1 Implementation

### Input Schema Design

First, define comprehensive input validation using Pydantic:

```python
from pydantic import BaseModel, Field
from typing import Optional

class GenerationInput(BaseModel):
    prompt: str
    height: int = Field(default=1024, ge=128, le=2048)
    width: int = Field(default=1024, ge=128, le=2048)
    num_inference_steps: int = Field(default=10, ge=1, le=30)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)

# Simplified input for basic usage
class MinifiedGenerationInput(BaseModel):
    prompt: str = "a beautiful mountain landscape"
```

### Custom Image Configuration

Create a pre-built image with the FLUX.1 model:

```python
from chutes.image import Image

# Create a markdown readme from model documentation
readme = """`FLUX.1 [dev]` is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.

# Key Features
1. Cutting-edge output quality, second only to our state-of-the-art model `FLUX.1 [pro]`.
2. Competitive prompt following, matching the performance of closed source alternatives.
3. Trained using guidance distillation, making `FLUX.1 [dev]` more efficient.
4. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
5. Generated outputs can be used for personal, scientific, and commercial purposes.
"""

# Use pre-built image with FLUX.1 model
image = (
    Image(
        username="myuser",
        name="flux.1-dev",
        tag="0.0.2",
        readme=readme)
    .from_base("parachutes/flux.1-dev:latest")
)
```

### Chute Configuration

Set up the service with appropriate hardware requirements:

```python
from chutes.chute import Chute, NodeSelector

chute = Chute(
    username="myuser",
    name="FLUX.1-dev-generator",
    readme=readme,
    image=image,
    # FLUX.1 requires significant GPU memory
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=80,  # 80GB for optimal performance
    ),
    # Limit concurrency due to memory requirements
    concurrency=1)
```

### Model Initialization

Initialize the diffusion pipeline on startup:

```python
@chute.on_startup()
async def initialize_pipeline(self):
    """
    Initialize the FLUX.1 pipeline with optimizations.
    """
    import torch
    from diffusers import FluxPipeline

    # Clear GPU cache and initialize
    self.torch = torch
    torch.cuda.empty_cache()
    torch.cuda.init()
    torch.cuda.set_device(0)

    # Load pre-downloaded model for faster startup
    self.pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True,  # Use cached model
        cache_dir="/home/chutes/.cache/huggingface/hub").to("cuda")
```

### Generation Endpoint

Create the main image generation endpoint:

```python
import uuid
from io import BytesIO
from fastapi import Response

@chute.cord(
    public_api_path="/generate",
    method="POST",
    input_schema=GenerationInput,
    minimal_input_schema=MinifiedGenerationInput,
    output_content_type="image/jpeg")
async def generate(self, params: GenerationInput) -> Response:
    """
    Generate high-quality images from text prompts.
    """
    # Set up random seed if provided
    generator = None
    if params.seed is not None:
        generator = self.torch.Generator(device="cuda").manual_seed(params.seed)

    # Generate image with optimized inference
    with self.torch.inference_mode():
        result = self.pipeline(
            prompt=params.prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            max_sequence_length=256,
            generator=generator)

    # Convert to JPEG and return
    image = result.images[0]
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{uuid.uuid4()}.jpg"'
        })
```

## Alternative: Stable Diffusion Setup

For a more customizable approach using Stable Diffusion:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

# Build custom Stable Diffusion image
image = (
    Image(username="myuser", name="stable-diffusion", tag="2.1")
    .from_base("nvidia/cuda:11.8-devel-ubuntu22.04")
    .run_command("apt update && apt install -y python3 python3-pip git")
    .run_command("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    .run_command("pip3 install diffusers transformers accelerate")
    .run_command("pip3 install fastapi uvicorn pydantic pillow")
    .set_workdir("/app")
)

chute = Chute(
    username="myuser",
    name="stable-diffusion-xl",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24),
    concurrency=2)

@chute.on_startup()
async def load_sd_pipeline(self):
    """Load Stable Diffusion XL pipeline."""
    from diffusers import StableDiffusionXLPipeline
    import torch

    self.pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True).to("cuda")

    # Enable memory efficient attention
    self.pipe.enable_memory_efficient_attention()

@chute.cord(public_api_path="/sdxl", method="POST")
async def generate_sdxl(self, prompt: str, width: int = 1024, height: int = 1024):
    """Generate images with Stable Diffusion XL."""
    images = self.pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=20).images

    # Return first image as base64
    buffer = BytesIO()
    images[0].save(buffer, format="PNG")
    return {"image": base64.b64encode(buffer.getvalue()).decode()}
```

## Advanced Features

### Batch Generation

Generate multiple images in a single request:

```python
from typing import List

class BatchGenerationInput(BaseModel):
    prompts: List[str] = Field(max_items=4)  # Limit batch size
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    num_inference_steps: int = Field(default=20, ge=10, le=50)

@chute.cord(public_api_path="/batch", method="POST")
async def generate_batch(self, params: BatchGenerationInput) -> List[str]:
    """Generate multiple images from prompts."""
    results = []

    for prompt in params.prompts:
        with self.torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=params.num_inference_steps)

        # Convert to base64
        buffer = BytesIO()
        result.images[0].save(buffer, format="JPEG", quality=90)
        b64_image = base64.b64encode(buffer.getvalue()).decode()
        results.append(b64_image)

    return results
```

### Image-to-Image Generation

Transform existing images with text prompts:

```python
import base64
from PIL import Image as PILImage

class Img2ImgInput(BaseModel):
    prompt: str
    image_b64: str  # Base64 encoded input image
    strength: float = Field(default=0.75, ge=0.1, le=1.0)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)

@chute.cord(public_api_path="/img2img", method="POST")
async def image_to_image(self, params: Img2ImgInput) -> Response:
    """Transform images with text prompts."""
    # Decode input image
    image_data = base64.b64decode(params.image_b64)
    init_image = PILImage.open(BytesIO(image_data)).convert("RGB")

    # Generate transformed image
    with self.torch.inference_mode():
        result = self.pipeline(
            prompt=params.prompt,
            image=init_image,
            strength=params.strength,
            guidance_scale=params.guidance_scale)

    # Return as JPEG
    buffer = BytesIO()
    result.images[0].save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg")
```

### Inpainting Support

Fill or edit specific regions of images:

```python
class InpaintInput(BaseModel):
    prompt: str
    image_b64: str      # Original image
    mask_b64: str       # Mask (white = inpaint, black = keep)
    strength: float = Field(default=0.75, ge=0.1, le=1.0)

@chute.on_startup()
async def load_inpaint_pipeline(self):
    """Load inpainting-specific pipeline."""
    from diffusers import StableDiffusionInpaintPipeline

    self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16).to("cuda")

@chute.cord(public_api_path="/inpaint", method="POST")
async def inpaint(self, params: InpaintInput) -> Response:
    """Inpaint regions of images."""
    # Decode images
    image_data = base64.b64decode(params.image_b64)
    mask_data = base64.b64decode(params.mask_b64)

    image = PILImage.open(BytesIO(image_data)).convert("RGB")
    mask = PILImage.open(BytesIO(mask_data)).convert("L")

    # Generate inpainted result
    result = self.inpaint_pipe(
        prompt=params.prompt,
        image=image,
        mask_image=mask,
        strength=params.strength)

    # Return result
    buffer = BytesIO()
    result.images[0].save(buffer, format="PNG")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")
```

## Deployment and Usage

### Deploy Your Service

```bash
# Build and deploy the image generation service
chutes deploy my_image_gen:chute

# Monitor deployment status
chutes chutes get my-image-gen
```

### Using the API

#### Basic Generation

```bash
curl -X POST "https://myuser-my-image-gen.chutes.ai/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic dragon flying over a crystal lake at sunset",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }' \
  --output generated_image.jpg
```

#### Python Client

```python
import requests
import base64
from PIL import Image
from io import BytesIO

def generate_image(prompt, **kwargs):
    """Generate image using your Chutes service."""
    url = "https://myuser-my-image-gen.chutes.ai/generate"

    payload = {
        "prompt": prompt,
        **kwargs
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        # Save image
        with open("generated.jpg", "wb") as f:
            f.write(response.content)

        # Or display in Jupyter
        image = Image.open(BytesIO(response.content))
        return image
    else:
        print(f"Error: {response.status_code}")
        return None

# Generate an image
image = generate_image(
    "a cyberpunk cityscape with neon lights and flying cars",
    width=1920,
    height=1080,
    num_inference_steps=25,
    seed=123
)
```

## Performance Optimization

### Memory Management

```python
# Enable memory efficient attention
self.pipeline.enable_memory_efficient_attention()

# Use attention slicing for large images
self.pipeline.enable_attention_slicing()

# Enable CPU offloading for very large models
self.pipeline.enable_model_cpu_offload()
```

### Speed Optimizations

```python
# Compile the UNet for faster inference
self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")

# Use faster schedulers
from diffusers import DPMSolverMultistepScheduler
self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    self.pipeline.scheduler.config
)
```

### Hardware Scaling

```python
# Scale up for higher throughput
node_selector = NodeSelector(
    gpu_count=2,  # Multi-GPU setup
    min_vram_gb_per_gpu=40)

# Or scale out with multiple instances
chute = Chute(
    # ... configuration
    concurrency=4,  # Handle more concurrent requests
)
```

## Best Practices

### 1. Prompt Engineering

```python
# Good prompts are specific and detailed
good_prompt = """
a photorealistic portrait of a wise old wizard with a long white beard,
wearing a starry blue robe, holding a glowing crystal staff,
in a mystical forest clearing with soft golden sunlight filtering through trees,
highly detailed, 8k resolution, fantasy art style
"""

# Add negative prompts to avoid unwanted elements
negative_prompt = """
blurry, low quality, deformed, ugly, bad anatomy,
watermark, signature, text, cropped
"""
```

### 2. Parameter Tuning

```python
# High quality settings
high_quality_params = {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
}

# Fast generation settings
fast_params = {
    "num_inference_steps": 15,
    "guidance_scale": 5.0,
    "width": 512,
    "height": 512,
}
```

### 3. Error Handling

```python
@chute.cord(public_api_path="/generate", method="POST")
async def generate_with_fallback(self, params: GenerationInput) -> Response:
    """Generate with proper error handling."""
    try:
        # Try high-quality generation first
        result = self.pipeline(
            prompt=params.prompt,
            width=params.width,
            height=params.height,
            num_inference_steps=params.num_inference_steps)

    except torch.cuda.OutOfMemoryError:
        # Fallback to lower resolution
        logger.warning("OOM error, reducing resolution")
        result = self.pipeline(
            prompt=params.prompt,
            width=params.width // 2,
            height=params.height // 2,
            num_inference_steps=params.num_inference_steps // 2)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

    # Return image...
```

## Monitoring and Scaling

### Resource Monitoring

```bash
# Check GPU utilization
chutes chutes metrics my-image-gen

# View generation logs
chutes chutes logs my-image-gen --tail 100

# Monitor request patterns
chutes chutes status my-image-gen
```

### Auto-scaling Configuration

```python
# Configure auto-scaling based on queue length
chute = Chute(
    # ... other config
    concurrency=2,           # Base concurrency
    max_replicas=5,          # Scale up to 5 instances
    scale_up_threshold=10,   # Scale when queue > 10
    scale_down_delay=300,    # Wait 5 min before scaling down
)
```

## Next Steps

- **Advanced Models**: Experiment with ControlNet, LoRA fine-tuning
- **Custom Training**: Train models on your own datasets
- **Integration**: Build web interfaces and mobile apps
- **Optimization**: Implement caching and CDN distribution

For more advanced examples, see:

- [Video Generation](/docs/examples/video-generation)
- [Custom Images](/docs/examples/custom-images)
- [Streaming Responses](/docs/examples/streaming-responses)
