# Video Generation with Wan2.1

This guide demonstrates how to build a sophisticated video generation service using Wan2.1-14B from Alibaba, capable of generating high-quality videos from text prompts and transforming images into videos.

## Overview

Wan2.1-14B is a state-of-the-art video generation model that supports:

- **Text-to-Video (T2V)**: Generate videos from text descriptions
- **Image-to-Video (I2V)**: Transform images into dynamic videos
- **Text-to-Image (T2I)**: Generate single frames from text
- **Multiple Resolutions**: Support for various aspect ratios
- **High Quality**: Up to 720p video generation with 44.1kHz audio
- **Distributed Processing**: Multi-GPU support for large-scale deployment

## Complete Implementation

### Input Schema Design

Define comprehensive input validation for video generation:

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Resolution(str, Enum):
    SIXTEEN_NINE = "1280*720"    # 16:9 widescreen
    NINE_SIXTEEN = "720*1280"    # 9:16 portrait (mobile)
    WIDESCREEN = "832*480"       # Cinematic widescreen
    PORTRAIT = "480*832"         # Portrait
    SQUARE = "1024*1024"         # Square format

class VideoGenInput(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = (
        "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, "
        "painting, picture, still, overall grayish, worst quality, low quality, JPEG compression artifacts, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, "
        "malformed limbs, fused fingers, motionless image, cluttered background, three legs, "
        "many people in the background, walking backwards, slow motion"
    )
    resolution: Optional[Resolution] = Resolution.WIDESCREEN
    sample_shift: Optional[float] = Field(None, ge=1.0, le=7.0)
    guidance_scale: Optional[float] = Field(5.0, ge=1.0, le=7.5)
    seed: Optional[int] = 42
    steps: int = Field(25, ge=10, le=30)
    fps: int = Field(16, ge=16, le=60)
    frames: Optional[int] = Field(81, ge=81, le=241)
    single_frame: Optional[bool] = False

class ImageGenInput(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = (
        "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, "
        "painting, picture, still, overall grayish, worst quality, low quality, JPEG compression artifacts, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, "
        "malformed limbs, fused fingers, motionless image, cluttered background, three legs, "
        "many people in the background, walking backwards, slow motion"
    )
    resolution: Optional[Resolution] = Resolution.WIDESCREEN
    sample_shift: Optional[float] = Field(None, ge=1.0, le=7.0)
    guidance_scale: Optional[float] = Field(5.0, ge=1.0, le=7.5)
    seed: Optional[int] = 42

class I2VInput(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = (
        "Vibrant colors, overexposed, static, blurry details, subtitles, style, artwork, "
        "painting, picture, still, overall grayish, worst quality, low quality, JPEG compression artifacts, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, "
        "malformed limbs, fused fingers, motionless image, cluttered background, three legs, "
        "many people in the background, walking backwards, slow motion"
    )
    sample_shift: Optional[float] = Field(None, ge=1.0, le=7.0)
    guidance_scale: Optional[float] = Field(5.0, ge=1.0, le=7.5)
    seed: Optional[int] = 42
    image_b64: str  # Base64 encoded input image
    steps: int = Field(25, ge=20, le=50)
    fps: int = Field(16, ge=16, le=60)
    single_frame: Optional[bool] = False
```

### Custom Image with Wan2.1

Build a custom image with all required dependencies:

```python
from chutes.image import Image as ChutesImage
from chutes.chute import Chute, NodeSelector
import os
import time
from loguru import logger

# Set up environment for large model downloads
T2V_PATH = os.path.join(os.getenv("HF_HOME", "/cache"), "Wan2.1-T2V-14B")
I2V_480_PATH = os.path.join(os.getenv("HF_HOME", "/cache"), "Wan2.1-I2V-14B-480P")

# Download models if in remote execution context
if os.getenv("CHUTES_EXECUTION_CONTEXT") == "REMOTE":
    from huggingface_hub import snapshot_download

    cache_dir = os.getenv("HF_HOME", "/cache")
    for _ in range(3):  # Retry downloads
        try:
            snapshot_download(
                repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                revision="6b73f84e66371cdfe870c72acd6826e1d61cf279",
                local_dir=I2V_480_PATH)
            snapshot_download(
                repo_id="Wan-AI/Wan2.1-T2V-14B",
                revision="b1cbf2d3d13dca5164463128885ab8e551e93e40",
                local_dir=T2V_PATH)
            break
        except Exception as exc:
            logger.warning(f"Error downloading models: {exc}")
            time.sleep(30)

# Build custom image with video generation capabilities
image = (
    ChutesImage(
        username="myuser",
        name="wan21",
        tag="0.0.1",
        readme="## Video and image generation/editing model from Alibaba")
    .from_base("parachutes/base-python:3.12.7")
    .set_user("root")
    .run_command("apt update")
    .apt_install("ffmpeg")  # Required for video processing
    .set_user("chutes")
    .run_command(
        "git clone https://github.com/Wan-Video/Wan2.1 && "
        "cd Wan2.1 && "
        "pip install --upgrade pip && "
        "pip install setuptools wheel && "
        "pip install torch torchvision && "
        "pip install -r requirements.txt --no-build-isolation && "
        "pip install xfuser && "
        # Apply critical patches for performance
        "perl -pi -e 's/sharding_strategy=sharding_strategy,/sharding_strategy=sharding_strategy,\\n        use_orig_params=True,/g' wan/distributed/fsdp.py && "
        "perl -pi -e 's/dtype=torch.float32,/dtype=torch.bfloat16,/g' wan/modules/t5.py && "
        "mv -f /app/Wan2.1/wan /home/chutes/.local/lib/python3.12/site-packages/"
    )
)
```

### Chute Configuration

Configure the service with high-end GPU requirements:

```python
chute = Chute(
    username="myuser",
    name="wan2.1-14b",
    tagline="Text-to-video, image-to-video, text-to-image with Wan2.1 14B",
    readme="High-quality video generation using Wan2.1 14B model with support for multiple formats and resolutions",
    image=image,
    node_selector=NodeSelector(
        gpu_count=8,  # Multi-GPU setup required
        include=["h100", "h800", "h100_nvl", "h100_sxm", "h200"]  # Latest GPUs only
    ))
```

### Distributed Model Initialization

Initialize models across multiple GPUs using distributed processing:

```python
def initialize_model(rank, world_size, task_queue):
    """
    Initialize Wan2.1 models in distributed fashion across GPUs.
    """
    import torch
    import torch.distributed as dist
    import wan
    from wan.configs import WAN_CONFIGS
    from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

    # Set up distributed environment
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    local_rank = rank
    device = local_rank
    torch.cuda.set_device(local_rank)

    logger.info(f"Initializing distributed inference on {rank=}...")
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29501",
        rank=rank,
        world_size=world_size
    )

    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=8)

    # Initialize text-to-video model
    cfg = WAN_CONFIGS["t2v-14B"]
    base_seed = [42] if rank == 0 else [None]
    dist.broadcast_object_list(base_seed, src=0)

    logger.info(f"Loading text-to-video model on {rank=}")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=T2V_PATH,
        device_id=device,
        rank=rank,
        t5_fsdp=True,
        dit_fsdp=True,
        use_usp=True,
        t5_cpu=False)

    # Compile models for optimal performance
    logger.info("Compiling text-to-video model...")
    wan_t2v.text_encoder = torch.compile(wan_t2v.text_encoder)
    wan_t2v.vae.model = torch.compile(wan_t2v.vae.model)
    wan_t2v.model = torch.compile(wan_t2v.model)

    # Initialize image-to-video model
    logger.info(f"Loading 480P image-to-video model on {rank=}")
    cfg = WAN_CONFIGS["i2v-14B"]
    wan_i2v_480 = wan.WanI2V(
        config=cfg,
        checkpoint_dir=I2V_480_PATH,
        device_id=device,
        rank=rank,
        t5_fsdp=True,
        dit_fsdp=True,
        use_usp=True,
        t5_cpu=False)

    logger.info("Compiling 480P image-to-video model...")
    wan_i2v_480.text_encoder = torch.compile(wan_i2v_480.text_encoder)
    wan_i2v_480.vae.model = torch.compile(wan_i2v_480.vae.model)
    wan_i2v_480.model = torch.compile(wan_i2v_480.model)

    logger.info(f"Finished loading models on {rank=}")

    if rank == 0:
        return wan_t2v, wan_i2v_480
    else:
        # Worker processes handle task queue
        while True:
            task = task_queue.get()
            prompt = task.get("prompt")
            args = task.get("args")

            if task.get("type") == "T2V":
                logger.info(f"Process {rank} executing T2V task...")
                _ = wan_t2v.generate(prompt, **args)
            else:  # I2V task
                logger.info(f"Process {rank} executing I2V 480P task...")
                _ = wan_i2v_480.generate(prompt, task["image"], **args)

            dist.barrier()

@chute.on_startup()
async def initialize(self):
    """
    Initialize distributed video generation system.
    """
    import torch
    import torch.multiprocessing as torch_mp
    import multiprocessing
    import numpy as np
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS
    from PIL import Image

    start_time = int(time.time())
    self.world_size = torch.cuda.device_count()
    torch_mp.set_start_method("spawn", force=True)

    # Create task queue for distributed processing
    processes = []
    self.task_queue = multiprocessing.Queue()

    logger.info(f"Starting {self.world_size} processes for distributed execution...")

    # Start worker processes
    for rank in range(1, self.world_size):
        p = torch_mp.Process(
            target=initialize_model,
            args=(rank, self.world_size, self.task_queue)
        )
        p.start()
        processes.append(p)
    self.processes = processes

    # Initialize main process models
    self.wan_t2v, self.wan_i2v_480 = initialize_model(0, self.world_size, self.task_queue)

    delta = int(time.time()) - start_time
    logger.success(f"Initialized T2V and I2V models in {delta} seconds!")

    # Perform warmup generations
    await self._warmup_models()

async def _warmup_models(self):
    """Warmup both T2V and I2V models with test generations."""
    import numpy as np
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS

    # Create synthetic warmup image
    array = np.zeros((480, 832, 3), dtype=np.uint8)
    for x in range(832):
        for y in range(480):
            r = int(255 * x / 832)
            g = int(255 * y / 480)
            b = int(255 * (x + y) / (832 + 480))
            array[y, x] = [r, g, b]
    warmup_image = Image.fromarray(array)

    # Warmup I2V model
    prompt_args = {
        "max_area": MAX_AREA_CONFIGS[Resolution.WIDESCREEN.value],
        "frame_num": 81,
        "shift": 3.0,
        "sample_solver": "unipc",
        "sampling_steps": 25,
        "guide_scale": 5.0,
        "seed": 42,
        "offload_model": False,
    }
    logger.info("Warming up image-to-video model...")
    _infer(self, "Shifting gradient.", image=warmup_image, single_frame=False, **prompt_args)

    # Warmup T2V model for all resolutions
    for resolution in (
        Resolution.SIXTEEN_NINE,
        Resolution.NINE_SIXTEEN,
        Resolution.WIDESCREEN,
        Resolution.PORTRAIT,
        Resolution.SQUARE):
        prompt_args = {
            "size": SIZE_CONFIGS[resolution.value],
            "frame_num": 81,
            "shift": 5.0,
            "sample_solver": "unipc",
            "sampling_steps": 25,
            "guide_scale": 5.0,
            "seed": 42,
            "offload_model": False,
        }
        logger.info(f"Warming up text-to-video model with {resolution=}")
        _infer(self, "a goat jumping off a boat", image=None, single_frame=False, **prompt_args)
```

### Core Inference Function

Create the unified inference function for all generation types:

```python
def _infer(self, prompt, image=None, single_frame=False, **prompt_args):
    """
    Unified inference function for T2V, I2V, and T2I generation.
    """
    import torch.distributed as dist
    from wan.utils.utils import cache_video, cache_image
    import uuid
    from io import BytesIO
    from fastapi import Response

    # Determine task type
    task_type = "I2V" if image else "T2V"
    if task_type == "I2V":
        _, height = image.size
        task_type += f"_{height}"

    # Distribute task to worker processes
    for _ in range(self.world_size - 1):
        self.task_queue.put({
            "type": task_type,
            "prompt": prompt,
            "image": image,
            "args": prompt_args
        })

    # Generate on main process
    model = getattr(self, f"wan_{task_type.lower()}")
    if image:
        video = model.generate(prompt, image, **prompt_args)
    else:
        video = model.generate(prompt, **prompt_args)

    # Wait for all processes to complete
    dist.barrier()

    # Save result (only on rank 0)
    if os.getenv("RANK") == "0":
        extension = "png" if single_frame else "mp4"
        output_file = f"/tmp/{uuid.uuid4()}.{extension}"

        try:
            if single_frame:
                output_file = cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=output_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                output_file = cache_video(
                    tensor=video[None],
                    save_file=output_file,
                    fps=prompt_args.get("fps", 16),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))

            if not output_file:
                raise Exception("Failed to save output!")

            # Read file and return response
            buffer = BytesIO()
            with open(output_file, "rb") as infile:
                buffer.write(infile.read())
            buffer.seek(0)

            media_type = "video/mp4" if not single_frame else "image/png"
            return Response(
                content=buffer.getvalue(),
                media_type=media_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{uuid.uuid4()}.{extension}"'
                })

        finally:
            if output_file and os.path.exists(output_file):
                os.remove(output_file)
```

### Video Generation Endpoints

Create endpoints for different generation modes:

```python
import base64
from io import BytesIO
from PIL import Image
from fastapi import HTTPException, status

@chute.cord(
    public_api_path="/text2video",
    public_api_method="POST",
    stream=False,
    output_content_type="video/mp4")
async def text_to_video(self, args: VideoGenInput):
    """
    Generate video from text description.
    """
    from wan.configs import SIZE_CONFIGS

    if args.sample_shift is None:
        args.sample_shift = 5.0

    if args.single_frame:
        args.frames = 1
    elif args.frames % 4 != 1:
        # Ensure frame count is compatible
        args.frames = args.frames - (args.frames % 4) + 1

    if not args.frames:
        args.frames = 81

    prompt_args = {
        "size": SIZE_CONFIGS[args.resolution.value],
        "frame_num": args.frames,
        "shift": args.sample_shift,
        "sample_solver": "unipc",
        "sampling_steps": args.steps,
        "guide_scale": args.guidance_scale,
        "seed": args.seed,
        "offload_model": False,
    }

    return _infer(
        self,
        args.prompt,
        image=None,
        single_frame=args.single_frame,
        **prompt_args
    )

@chute.cord(
    public_api_path="/text2image",
    public_api_method="POST",
    stream=False,
    output_content_type="image/png")
async def text_to_image(self, args: ImageGenInput):
    """
    Generate single image from text description.
    """
    # Convert to video input with single frame
    vargs = VideoGenInput(**args.model_dump())
    vargs.single_frame = True
    return await text_to_video(self, vargs)

def prepare_input_image(args):
    """
    Resize and crop input image to target resolution.
    """
    target_width = 832
    target_height = 480

    try:
        input_image = Image.open(BytesIO(base64.b64decode(args.image_b64)))
        orig_width, orig_height = input_image.size

        # Calculate scaling to maintain aspect ratio
        width_ratio = target_width / orig_width
        height_ratio = target_height / orig_height
        scale_factor = max(width_ratio, height_ratio)

        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)

        # Resize image
        input_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to target dimensions
        width, height = input_image.size
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        input_image = input_image.crop((left, top, right, bottom)).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image input! {exc}")

    return input_image

@chute.cord(
    public_api_path="/image2video",
    public_api_method="POST",
    stream=False,
    output_content_type="video/mp4")
async def image_to_video(self, args: I2VInput):
    """
    Generate video from input image and text prompt.
    """
    from wan.configs import MAX_AREA_CONFIGS

    if args.sample_shift is None:
        args.sample_shift = 3.0

    # Process and validate input image
    input_image = prepare_input_image(args)

    prompt_args = {
        "max_area": MAX_AREA_CONFIGS[Resolution.WIDESCREEN.value],
        "frame_num": 81,  # Fixed frame count for stability
        "shift": args.sample_shift,
        "sample_solver": "unipc",
        "sampling_steps": args.steps,
        "guide_scale": args.guidance_scale,
        "seed": args.seed,
        "offload_model": False,
    }

    return _infer(
        self,
        args.prompt,
        image=input_image,
        single_frame=False,
        **prompt_args
    )
```

## Advanced Features

### Batch Video Generation

Process multiple prompts efficiently:

```python
class BatchVideoInput(BaseModel):
    prompts: List[str] = Field(max_items=5)  # Limit for resource management
    resolution: Resolution = Resolution.WIDESCREEN
    steps: int = Field(20, ge=10, le=30)
    frames: int = Field(81, ge=81, le=161)

@chute.cord(public_api_path="/batch_video", method="POST")
async def batch_video_generation(self, args: BatchVideoInput) -> List[str]:
    """Generate multiple videos and return as base64 list."""
    from wan.configs import SIZE_CONFIGS

    results = []

    for prompt in args.prompts:
        prompt_args = {
            "size": SIZE_CONFIGS[args.resolution.value],
            "frame_num": args.frames,
            "shift": 5.0,
            "sample_solver": "unipc",
            "sampling_steps": args.steps,
            "guide_scale": 5.0,
            "seed": 42,
            "offload_model": False,
        }

        response = _infer(self, prompt, image=None, single_frame=False, **prompt_args)

        # Convert response to base64
        video_b64 = base64.b64encode(response.body).decode()
        results.append(video_b64)

    return results
```

### Style-Guided Video Generation

Add style control to video generation:

```python
class StyledVideoInput(BaseModel):
    prompt: str
    style: str = "cinematic"  # Style guidance
    mood: str = "dramatic"    # Mood control
    camera_movement: str = "static"  # Camera motion
    resolution: Resolution = Resolution.WIDESCREEN
    steps: int = Field(25, ge=15, le=35)

@chute.cord(public_api_path="/styled_video", method="POST")
async def styled_video_generation(self, args: StyledVideoInput) -> Response:
    """Generate video with style and mood control."""

    # Enhance prompt with style guidance
    enhanced_prompt = f"{args.prompt}, {args.style} style, {args.mood} mood"
    if args.camera_movement != "static":
        enhanced_prompt += f", {args.camera_movement} camera movement"

    # Generate with enhanced prompt
    video_args = VideoGenInput(
        prompt=enhanced_prompt,
        resolution=args.resolution,
        steps=args.steps,
        frames=81,
        single_frame=False
    )

    return await text_to_video(self, video_args)
```

### Video Interpolation

Create smooth transitions between keyframes:

```python
class InterpolationInput(BaseModel):
    start_prompt: str
    end_prompt: str
    interpolation_steps: int = Field(5, ge=3, le=10)
    resolution: Resolution = Resolution.WIDESCREEN

@chute.cord(public_api_path="/interpolate_video", method="POST")
async def video_interpolation(self, args: InterpolationInput) -> Response:
    """Generate video that interpolates between two prompts."""

    # Generate interpolated prompts
    interpolated_prompts = []
    for i in range(args.interpolation_steps):
        weight = i / (args.interpolation_steps - 1)

        if weight == 0:
            prompt = args.start_prompt
        elif weight == 1:
            prompt = args.end_prompt
        else:
            # Simple linear interpolation in text space
            prompt = f"transitioning from {args.start_prompt} to {args.end_prompt}, step {i+1}"

        interpolated_prompts.append(prompt)

    # Generate sequence of videos
    video_segments = []
    for prompt in interpolated_prompts:
        video_args = VideoGenInput(
            prompt=prompt,
            resolution=args.resolution,
            frames=41,  # Shorter segments for smooth transition
            steps=20
        )

        segment_response = await text_to_video(self, video_args)
        video_segments.append(segment_response.body)

    # Concatenate videos (simplified - would need ffmpeg for production)
    # For now, return the last segment
    return Response(
        content=video_segments[-1],
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=interpolated_video.mp4"}
    )
```

## Deployment and Usage

### Deploy the Service

```bash
# Build and deploy the video generation service
chutes deploy my_video_gen:chute

# Monitor the deployment (this will take time due to model size)
chutes chutes get my-video-gen
```

### Using the API

#### Text-to-Video Generation

```bash
curl -X POST "https://myuser-my-video-gen.chutes.ai/text2video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a majestic eagle soaring over mountain peaks at golden hour",
    "resolution": "1280*720",
    "steps": 25,
    "frames": 81,
    "fps": 24,
    "seed": 12345
  }' \
  --output eagle_video.mp4
```

#### Image-to-Video Generation

```bash
# First encode your image to base64
base64 -i input_image.jpg > image.b64

curl -X POST "https://myuser-my-video-gen.chutes.ai/image2video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "gentle waves lapping against the shore",
    "image_b64": "'$(cat image.b64)'",
    "steps": 30,
    "fps": 16,
    "seed": 42
  }' \
  --output animated_image.mp4
```

#### Python Client Example

```python
import requests
import base64
from typing import List, Optional
from enum import Enum

class VideoGenerator:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def text_to_video(
        self,
        prompt: str,
        resolution: str = "832*480",
        steps: int = 25,
        frames: int = 81,
        fps: int = 16,
        seed: Optional[int] = None
    ) -> bytes:
        """Generate video from text prompt."""

        payload = {
            "prompt": prompt,
            "resolution": resolution,
            "steps": steps,
            "frames": frames,
            "fps": fps,
            "single_frame": False
        }

        if seed is not None:
            payload["seed"] = seed

        response = requests.post(
            f"{self.base_url}/text2video",
            json=payload,
            timeout=300  # Extended timeout for video generation
        )

        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Video generation failed: {response.status_code} - {response.text}")

    def image_to_video(
        self,
        prompt: str,
        image_path: str,
        steps: int = 25,
        fps: int = 16,
        seed: Optional[int] = None
    ) -> bytes:
        """Generate video from image and text prompt."""

        # Encode image to base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "prompt": prompt,
            "image_b64": image_b64,
            "steps": steps,
            "fps": fps,
            "single_frame": False
        }

        if seed is not None:
            payload["seed"] = seed

        response = requests.post(
            f"{self.base_url}/image2video",
            json=payload,
            timeout=300
        )

        return response.content

    def text_to_image(
        self,
        prompt: str,
        resolution: str = "1024*1024",
        seed: Optional[int] = None
    ) -> bytes:
        """Generate single image from text."""

        payload = {
            "prompt": prompt,
            "resolution": resolution
        }

        if seed is not None:
            payload["seed"] = seed

        response = requests.post(
            f"{self.base_url}/text2image",
            json=payload,
            timeout=120
        )

        return response.content

    def styled_video(
        self,
        prompt: str,
        style: str = "cinematic",
        mood: str = "dramatic",
        camera_movement: str = "static"
    ) -> bytes:
        """Generate styled video."""

        payload = {
            "prompt": prompt,
            "style": style,
            "mood": mood,
            "camera_movement": camera_movement,
            "resolution": "1280*720",
            "steps": 25
        }

        response = requests.post(
            f"{self.base_url}/styled_video",
            json=payload,
            timeout=300
        )

        return response.content

# Usage examples
generator = VideoGenerator("https://myuser-my-video-gen.chutes.ai")

# Generate cinematic video
video = generator.text_to_video(
    prompt="A time-lapse of a bustling city street transitioning from day to night",
    resolution="1280*720",
    frames=121,
    fps=24,
    seed=12345
)

with open("city_timelapse.mp4", "wb") as f:
    f.write(video)

# Animate a photograph
animated = generator.image_to_video(
    prompt="gentle autumn breeze causing leaves to fall",
    image_path="autumn_scene.jpg",
    steps=30,
    fps=16
)

with open("animated_autumn.mp4", "wb") as f:
    f.write(animated)

# Generate styled content
styled_video = generator.styled_video(
    prompt="a lone warrior walking through a desert",
    style="epic fantasy",
    mood="heroic",
    camera_movement="slow pan"
)

with open("epic_warrior.mp4", "wb") as f:
    f.write(styled_video)
```

## Performance Optimization

### Memory Management

The model requires significant GPU memory and careful management:

```python
# Monitor and optimize memory usage
@chute.cord(public_api_path="/optimized_video", method="POST")
async def optimized_video_generation(self, args: VideoGenInput) -> Response:
    """Memory-optimized video generation."""
    import torch

    try:
        # Clear cache before generation
        torch.cuda.empty_cache()

        # Reduce frame count for memory efficiency if needed
        if args.frames > 161:
            args.frames = 161
            logger.warning("Reduced frame count for memory efficiency")

        # Generate with memory monitoring
        result = await text_to_video(self, args)

        return result

    except torch.cuda.OutOfMemoryError:
        # Fallback to lower resolution/frame count
        logger.warning("OOM detected, falling back to lower settings")

        args.resolution = Resolution.WIDESCREEN  # Smaller resolution
        args.frames = 81  # Fewer frames

        torch.cuda.empty_cache()
        return await text_to_video(self, args)

    finally:
        # Always clean up
        torch.cuda.empty_cache()
```

### Quality vs Speed Trade-offs

```python
class QualityPreset(str, Enum):
    FAST = "fast"        # 15 steps, 720p, 81 frames
    BALANCED = "balanced" # 25 steps, 1080p, 121 frames
    QUALITY = "quality"   # 35 steps, 1080p, 161 frames

@chute.cord(public_api_path="/preset_video", method="POST")
async def preset_video_generation(self, prompt: str, preset: QualityPreset = QualityPreset.BALANCED) -> Response:
    """Generate video with quality presets."""

    if preset == QualityPreset.FAST:
        args = VideoGenInput(
            prompt=prompt,
            resolution=Resolution.WIDESCREEN,
            steps=15,
            frames=81,
            fps=16
        )
    elif preset == QualityPreset.BALANCED:
        args = VideoGenInput(
            prompt=prompt,
            resolution=Resolution.SIXTEEN_NINE,
            steps=25,
            frames=121,
            fps=24
        )
    else:  # QUALITY
        args = VideoGenInput(
            prompt=prompt,
            resolution=Resolution.SIXTEEN_NINE,
            steps=35,
            frames=161,
            fps=30
        )

    return await text_to_video(self, args)
```

## Best Practices

### 1. Prompt Engineering for Video

```python
# Effective video prompts include motion and temporal elements
good_video_prompts = [
    "a cat gracefully leaping from a windowsill to a nearby table",
    "ocean waves gently rolling onto a sandy beach at sunset",
    "time-lapse of cherry blossoms blooming in spring",
    "a paper airplane gliding through the air in slow motion",
    "raindrops creating ripples on a calm pond surface"
]

# Avoid static descriptions better suited for images
avoid_for_video = [
    "a beautiful mountain landscape",  # Too static
    "portrait of a person",           # No implied motion
    "a red car",                      # Lacks temporal context
]

# Add temporal and motion keywords
def enhance_video_prompt(base_prompt: str) -> str:
    """Enhance prompts for better video generation."""
    motion_words = [
        "flowing", "moving", "swaying", "drifting", "gliding",
        "rotating", "spinning", "floating", "cascading", "rippling"
    ]

    temporal_words = [
        "slowly", "gently", "gradually", "smoothly", "continuously",
        "rhythmically", "gracefully", "elegantly"
    ]

    # Simple enhancement (would be more sophisticated in practice)
    if not any(word in base_prompt.lower() for word in motion_words + temporal_words):
        return f"{base_prompt}, gently moving, smooth motion"

    return base_prompt
```

### 2. Resolution and Aspect Ratio Selection

```python
def select_optimal_resolution(content_type: str, platform: str = "web") -> Resolution:
    """Select optimal resolution based on content and platform."""

    if platform == "mobile":
        return Resolution.NINE_SIXTEEN  # Mobile-friendly portrait
    elif platform == "social":
        return Resolution.SQUARE        # Social media posts
    elif content_type == "cinematic":
        return Resolution.SIXTEEN_NINE  # Widescreen cinematic
    elif content_type == "portrait":
        return Resolution.PORTRAIT      # Portrait orientation
    else:
        return Resolution.WIDESCREEN    # General purpose
```

### 3. Error Handling and Fallbacks

```python
async def robust_video_generation(self, args: VideoGenInput) -> Response:
    """Generate video with comprehensive error handling and fallbacks."""

    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Validate input parameters
            if args.frames > 241:
                args.frames = 241

            if args.steps > 35:
                args.steps = 35

            # Generate video
            result = await text_to_video(self, args)
            return result

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM on attempt {attempt + 1}, reducing settings")

            # Progressive fallback strategy
            if attempt == 0:
                args.frames = min(args.frames, 121)  # Reduce frames
            elif attempt == 1:
                args.resolution = Resolution.WIDESCREEN  # Smaller resolution
                args.frames = 81
            else:
                args.steps = 15  # Faster generation
                args.frames = 41

            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Generation failed on attempt {attempt + 1}: {e}")

            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video generation failed after {max_retries} attempts"
                )

            time.sleep(5)  # Wait before retry
```

## Monitoring and Troubleshooting

### Resource Monitoring

```bash
# Monitor service health and resource usage
chutes chutes get my-video-gen

# View detailed logs
chutes chutes logs my-video-gen --tail 200

# Monitor GPU utilization across all devices
chutes chutes metrics my-video-gen --detailed
```

### Performance Metrics

```python
import time
from loguru import logger

@chute.cord(public_api_path="/monitored_video", method="POST")
async def monitored_video_generation(self, args: VideoGenInput) -> Response:
    """Video generation with performance monitoring."""

    start_time = time.time()
    gpu_memory_start = torch.cuda.memory_allocated()

    try:
        result = await text_to_video(self, args)

        generation_time = time.time() - start_time
        gpu_memory_peak = torch.cuda.max_memory_allocated()

        logger.info(
            f"Video generation completed - "
            f"Time: {generation_time:.2f}s, "
            f"Frames: {args.frames}, "
            f"Resolution: {args.resolution}, "
            f"GPU Memory: {gpu_memory_peak / 1024**3:.2f}GB"
        )

        return result

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Video generation failed after {error_time:.2f}s: {e}")
        raise

    finally:
        torch.cuda.reset_peak_memory_stats()
```

## Next Steps

- **Custom Training**: Fine-tune Wan2.1 on your own video datasets
- **Advanced Effects**: Implement video filters and post-processing
- **Real-time Streaming**: Build live video generation systems
- **Integration**: Connect with video editing and content creation tools

For more advanced examples, see:

- [Custom Training](/docs/examples/custom-training)
- [Streaming Applications](/docs/examples/streaming-responses)
- [Performance Optimization](/docs/examples/performance-optimization)
