# Diffusion Template

The **Diffusion template** provides high-performance image generation using Stable Diffusion and other diffusion models. Perfect for text-to-image, image-to-image, and inpainting applications.

## What is Stable Diffusion?

Stable Diffusion is a powerful diffusion model that generates high-quality images from text prompts:

- 🎨 **Text-to-image** generation from prompts
- 🖼️ **Image-to-image** transformation and editing
- 🎭 **Inpainting** to fill missing parts of images
- 🎯 **ControlNet** for guided generation
- ⚡ **Optimized inference** with multiple acceleration techniques

## Quick Start

```python
from chutes.chute import NodeSelector
from chutes.chute.template.diffusion import build_diffusion_chute

chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=12
    )
)
```

This creates a complete diffusion deployment with:

- ✅ Optimized Stable Diffusion pipeline
- ✅ Multiple generation modes (txt2img, img2img, inpaint)
- ✅ Configurable generation parameters
- ✅ Safety filtering and content moderation
- ✅ Auto-scaling based on demand

## Function Reference

### `build_diffusion_chute()`

```python
def build_diffusion_chute(
    username: str,
    model_name: str,
    revision: str = "main",
    node_selector: NodeSelector = None,
    image: str | Image = None,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 1,

    # Diffusion-specific parameters
    pipeline_type: str = "text2img",
    scheduler: str = "euler_a",
    safety_checker: bool = True,
    requires_safety_checker: bool = False,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512,
    enable_xformers: bool = True,
    enable_cpu_offload: bool = False,
    **kwargs
) -> Chute:
```

#### Required Parameters

- **`username`**: Your Chutes username
- **`model_name`**: HuggingFace diffusion model identifier

#### Diffusion Configuration

- **`pipeline_type`**: Generation mode - "text2img", "img2img", or "inpaint" (default: "text2img")
- **`scheduler`**: Sampling scheduler - "euler_a", "ddim", "dpm", etc. (default: "euler_a")
- **`safety_checker`**: Enable NSFW content filtering (default: True)
- **`guidance_scale`**: CFG guidance strength (default: 7.5)
- **`num_inference_steps`**: Number of denoising steps (default: 50)
- **`height`**: Default image height (default: 512)
- **`width`**: Default image width (default: 512)
- **`enable_xformers`**: Use memory-efficient attention (default: True)

## Complete Example

```python
from chutes.chute import NodeSelector
from chutes.chute.template.diffusion import build_diffusion_chute

# Build diffusion chute for image generation
chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,
        include=["rtx4090", "a100"]
    ),
    tagline="High-quality image generation with SDXL",
    readme="""
# Image Generation Service

Generate stunning images from text prompts using Stable Diffusion XL.

## Features
- High-resolution image generation (up to 1024x1024)
- Multiple generation modes
- ControlNet support for guided generation
- Safety filtering for appropriate content

## API Endpoints
- `/generate` - Text-to-image generation
- `/img2img` - Image-to-image transformation
- `/inpaint` - Image inpainting
    """,

    # Optimize for SDXL
    scheduler="euler_a",
    guidance_scale=7.5,
    num_inference_steps=30,  # SDXL works well with fewer steps
    height=1024,
    width=1024,
    safety_checker=True
)
```

## API Endpoints

### Text-to-Image Generation

```bash
curl -X POST https://myuser-diffusion-chute.chutes.ai/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape with mountains and a lake at sunset",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

### Image-to-Image

```bash
curl -X POST https://myuser-diffusion-chute.chutes.ai/img2img \
  -F "image=@input_image.jpg" \
  -F "prompt=A cyberpunk version of this scene" \
  -F "strength=0.7" \
  -F "guidance_scale=7.5"
```

### Inpainting

```bash
curl -X POST https://myuser-diffusion-chute.chutes.ai/inpaint \
  -F "image=@original.jpg" \
  -F "mask=@mask.jpg" \
  -F "prompt=A beautiful garden" \
  -F "num_inference_steps=50"
```

## Model Recommendations

### Stable Diffusion 1.5

```python
# Classic SD 1.5 - good balance of quality and speed
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8,
    include=["rtx3090", "rtx4090"]
)

# Recommended models:
# - runwayml/stable-diffusion-v1-5
# - stabilityai/stable-diffusion-2-1
# - prompthero/openjourney
```

### Stable Diffusion XL

```python
# SDXL - highest quality, more VRAM needed
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=12,
    include=["rtx4090", "a100"]
)

# Recommended models:
# - stabilityai/stable-diffusion-xl-base-1.0
# - stabilityai/stable-diffusion-xl-refiner-1.0
# - Lykon/DreamShaper-XL-1.0
```

### Specialized Models

```python
# Anime/artistic styles
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=10,
    include=["rtx4090", "a100"]
)

# Recommended models:
# - Linaqruf/anything-v3.0
# - hakurei/waifu-diffusion
# - SG161222/Realistic_Vision_V6.0_B1_noVAE
```

## Use Cases

### 1. **Marketing Content Creation**

```python
marketing_chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    tagline="Marketing image generation",
    guidance_scale=8.0,  # Higher guidance for consistent style
    num_inference_steps=40,
    height=1024,
    width=1024
)
```

### 2. **Art Generation**

```python
art_chute = build_diffusion_chute(
    username="myuser",
    model_name="Lykon/DreamShaper-XL-1.0",
    tagline="Artistic image creation",
    guidance_scale=6.0,  # Lower for more creative freedom
    scheduler="dpm_solver_multistep",
    safety_checker=False  # For artistic freedom
)
```

### 3. **Product Visualization**

```python
product_chute = build_diffusion_chute(
    username="myuser",
    model_name="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    tagline="Realistic product images",
    guidance_scale=7.5,
    num_inference_steps=50,  # More steps for photorealism
    scheduler="euler_a"
)
```

### 4. **Character Design**

```python
character_chute = build_diffusion_chute(
    username="myuser",
    model_name="Linaqruf/anything-v3.0",
    tagline="Character and concept art",
    guidance_scale=7.0,
    height=768,
    width=512  # Portrait orientation
)
```

## Advanced Features

### ControlNet Integration

```python
# Enable ControlNet for guided generation
controlnet_chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    enable_controlnet=True,
    controlnet_models=[
        "diffusers/controlnet-canny-sdxl-1.0",
        "diffusers/controlnet-depth-sdxl-1.0"
    ]
)
```

### Custom VAE

```python
# Use custom VAE for better image quality
custom_vae_chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    enable_vae_slicing=True
)
```

### Multi-Model Pipeline

```python
# SDXL with refiner for ultimate quality
refiner_chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    refiner_model="stabilityai/stable-diffusion-xl-refiner-1.0",
    refiner_strength=0.3,
    num_inference_steps=40
)
```

## Performance Optimization

### Speed Optimization

```python
# Optimize for fast generation
fast_chute = build_diffusion_chute(
    username="myuser",
    model_name="runwayml/stable-diffusion-v1-5",
    num_inference_steps=20,      # Fewer steps
    guidance_scale=5.0,          # Lower guidance
    enable_xformers=True,        # Memory efficient attention
    scheduler="euler_a",         # Fast scheduler
    enable_cpu_offload=False     # Keep everything on GPU
)
```

### Quality Optimization

```python
# Optimize for highest quality
quality_chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    num_inference_steps=50,      # More steps
    guidance_scale=8.0,          # Higher guidance
    scheduler="dpm_solver_multistep",  # High-quality scheduler
    height=1024,
    width=1024
)
```

### Memory Optimization

```python
# Optimize for lower VRAM usage
memory_efficient_chute = build_diffusion_chute(
    username="myuser",
    model_name="runwayml/stable-diffusion-v1-5",
    enable_cpu_offload=True,     # Offload to CPU when not in use
    enable_vae_slicing=True,     # Slice VAE for memory efficiency
    enable_attention_slicing=True, # Slice attention layers
    height=512,
    width=512
)
```

## Testing Your Diffusion Chute

### Python Client

```python
import requests
import base64
from PIL import Image
import io

def generate_image(prompt, negative_prompt="", width=1024, height=1024):
    """Generate image from text prompt."""

    response = requests.post(
        "https://myuser-diffusion-chute.chutes.ai/generate",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "seed": -1  # Random seed
        }
    )

    if response.status_code == 200:
        result = response.json()

        # Decode base64 image
        image_data = base64.b64decode(result["images"][0])
        image = Image.open(io.BytesIO(image_data))

        return image
    else:
        raise Exception(f"Generation failed: {response.text}")

# Test image generation
image = generate_image(
    prompt="A serene mountain lake at sunset with purple clouds",
    negative_prompt="blurry, low quality, distorted, text",
    width=1024,
    height=768
)

image.save("generated_image.png")
print("Image saved as generated_image.png")
```

### Batch Generation

```python
import asyncio
import aiohttp
import json

async def batch_generate_images(prompts):
    """Generate multiple images concurrently."""

    async def generate_single(session, prompt):
        async with session.post(
            "https://myuser-diffusion-chute.chutes.ai/generate",
            json={
                "prompt": prompt,
                "num_inference_steps": 25,
                "guidance_scale": 7.0,
                "width": 512,
                "height": 512
            }
        ) as response:
            return await response.json()

    async with aiohttp.ClientSession() as session:
        tasks = [generate_single(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

    return results

# Test batch generation
prompts = [
    "A majestic eagle soaring over mountains",
    "A cyberpunk cityscape at night with neon lights",
    "A peaceful garden with cherry blossoms",
    "A futuristic robot in a sci-fi laboratory"
]

results = asyncio.run(batch_generate_images(prompts))

for i, result in enumerate(results):
    print(f"Generated image {i+1} successfully")
```

### Image-to-Image Testing

```python
import requests
from PIL import Image

def img2img_transform(input_image_path, prompt, strength=0.7):
    """Transform an existing image based on prompt."""

    with open(input_image_path, 'rb') as f:
        files = {'image': f}
        data = {
            'prompt': prompt,
            'strength': strength,
            'guidance_scale': 7.5,
            'num_inference_steps': 30
        }

        response = requests.post(
            "https://myuser-diffusion-chute.chutes.ai/img2img",
            files=files,
            data=data
        )

    if response.status_code == 200:
        result = response.json()
        # Process result similar to text-to-image
        return result
    else:
        raise Exception(f"Transform failed: {response.text}")

# Test image transformation
result = img2img_transform(
    "input_photo.jpg",
    "Transform this into a watercolor painting",
    strength=0.8
)
```

## Generation Parameters Guide

### Prompt Engineering

```python
# Effective prompt structure
def create_effective_prompt(subject, style, quality_modifiers=""):
    """Create well-structured prompts."""

    base_prompt = f"{subject}, {style}"

    if quality_modifiers:
        base_prompt += f", {quality_modifiers}"

    # Add quality enhancers
    quality_terms = "highly detailed, sharp focus, professional photography"

    return f"{base_prompt}, {quality_terms}"

# Examples
portrait_prompt = create_effective_prompt(
    subject="Portrait of a young woman with curly hair",
    style="Renaissance painting style",
    quality_modifiers="oil painting, classical lighting"
)

landscape_prompt = create_effective_prompt(
    subject="Mountain landscape with a lake",
    style="digital art",
    quality_modifiers="golden hour lighting, cinematic composition"
)
```

### Parameter Guidelines

```python
# Parameter recommendations by use case

# Photorealistic images
photorealistic_params = {
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
    "scheduler": "euler_a"
}

# Artistic/creative images
artistic_params = {
    "guidance_scale": 6.0,
    "num_inference_steps": 30,
    "scheduler": "dpm_solver_multistep"
}

# Fast generation
fast_params = {
    "guidance_scale": 5.0,
    "num_inference_steps": 20,
    "scheduler": "euler_a"
}

# High quality (slow)
quality_params = {
    "guidance_scale": 8.5,
    "num_inference_steps": 80,
    "scheduler": "dpm_solver_multistep"
}
```

## Integration Examples

### Web Gallery Application

```python
from flask import Flask, request, jsonify, render_template
import requests
import base64

app = Flask(__name__)

@app.route('/')
def gallery():
    return render_template('gallery.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')

    # Generate image
    response = requests.post(
        "https://myuser-diffusion-chute.chutes.ai/generate",
        json={
            "prompt": prompt,
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "num_inference_steps": 25
        }
    )

    if response.status_code == 200:
        result = response.json()
        return jsonify({
            "success": True,
            "image": result["images"][0],  # Base64 encoded
            "seed": result.get("seed")
        })
    else:
        return jsonify({"success": False, "error": response.text})

if __name__ == '__main__':
    app.run(debug=True)
```

### Image Processing Pipeline

```python
import requests
from PIL import Image, ImageEnhance
import io
import base64

class ImageProcessor:
    def __init__(self, chute_url):
        self.chute_url = chute_url

    def generate_base_image(self, prompt):
        """Generate initial image."""
        response = requests.post(
            f"{self.chute_url}/generate",
            json={
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30
            }
        )

        result = response.json()
        image_data = base64.b64decode(result["images"][0])
        return Image.open(io.BytesIO(image_data))

    def refine_image(self, image, prompt, strength=0.5):
        """Refine existing image."""
        # Convert PIL image to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        files = {'image': img_buffer}
        data = {
            'prompt': prompt,
            'strength': strength,
            'num_inference_steps': 20
        }

        response = requests.post(
            f"{self.chute_url}/img2img",
            files=files,
            data=data
        )

        result = response.json()
        refined_data = base64.b64decode(result["images"][0])
        return Image.open(io.BytesIO(refined_data))

    def enhance_image(self, image):
        """Apply post-processing enhancements."""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)

        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)

        return image

# Usage example
processor = ImageProcessor("https://myuser-diffusion-chute.chutes.ai")

# Generate and refine
base_image = processor.generate_base_image("A beautiful sunset over the ocean")
refined_image = processor.refine_image(
    base_image,
    "A beautiful sunset over the ocean, cinematic lighting, golden hour",
    strength=0.3
)
final_image = processor.enhance_image(refined_image)

final_image.save("final_artwork.png")
```

## Troubleshooting

### Common Issues

**Generation too slow?**

- Reduce `num_inference_steps` (try 20-30)
- Use a faster scheduler like "euler_a"
- Lower the resolution (512x512 instead of 1024x1024)
- Enable memory optimizations

**Out of memory errors?**

- Enable CPU offloading: `enable_cpu_offload=True`
- Enable attention slicing: `enable_attention_slicing=True`
- Reduce image resolution
- Use a smaller model (SD 1.5 instead of SDXL)

**Poor image quality?**

- Increase `num_inference_steps` (try 50-80)
- Adjust `guidance_scale` (7.5-12.0)
- Improve prompts with quality modifiers
- Use a higher resolution

**NSFW content blocked?**

- Adjust prompts to be more appropriate
- Set `safety_checker=False` if appropriate for your use case
- Use different negative prompts

## Best Practices

### 1. **Prompt Engineering**

```python
# Good prompt structure
good_prompt = "Portrait of a person, photorealistic, highly detailed, professional photography, sharp focus, beautiful lighting"

# Include style modifiers
style_prompt = "Landscape painting, oil on canvas, Bob Ross style, happy little trees, peaceful, serene"

# Use negative prompts effectively
negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, text, watermark"
```

### 2. **Parameter Optimization**

```python
# Balance quality and speed
balanced_config = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "width": 768,
    "height": 768
}

# For batch processing
batch_config = {
    "num_inference_steps": 20,
    "guidance_scale": 6.0,
    "width": 512,
    "height": 512
}
```

### 3. **Memory Management**

```python
# For limited VRAM
memory_config = {
    "enable_cpu_offload": True,
    "enable_attention_slicing": True,
    "enable_vae_slicing": True,
    "width": 512,
    "height": 512
}
```

### 4. **Content Safety**

```python
# Enable safety checking for public-facing applications
safe_config = {
    "safety_checker": True,
    "requires_safety_checker": True,
    "guidance_scale": 7.5  # Moderate guidance
}
```

## Next Steps

- **[VLLM Template](/docs/templates/vllm)** - Text generation capabilities
- **[TEI Template](/docs/templates/tei)** - Text embeddings for image search
- **[Image Processing Guide](/docs/guides/image-processing)** - Advanced image manipulation
- **[ControlNet Guide](/docs/guides/controlnet)** - Guided image generation
