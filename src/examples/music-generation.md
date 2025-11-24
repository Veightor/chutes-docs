# Music Generation with DiffRhythm

This guide demonstrates how to build a sophisticated music generation service using DiffRhythm, capable of creating music from text prompts and lyrics with advanced rhythm and style control.

## Overview

DiffRhythm (ASLP-lab/DiffRhythm) is a state-of-the-art music generation model that can:

- Generate music from text descriptions and style prompts
- Convert lyrics with timing information into musical performances
- Use reference audio to guide musical style
- Support multiple languages and musical genres
- Generate high-quality 44.1kHz audio output

## Complete Implementation

### Input Schema Design

Define comprehensive input validation for music generation:

```python
import re
from typing import Optional
from pydantic import BaseModel
from fastapi import HTTPException, status

# Regex for validating LRC (lyric) format timestamps
LRC_RE = re.compile(r"\[(\d+):(\d+\.\d+)\]")

class InputArgs(BaseModel):
    style_prompt: Optional[str] = None
    lyrics: Optional[str] = None
    audio_b64: Optional[str] = None  # Reference audio in base64
```

### Custom Image with DiffRhythm

Build a custom image with all required dependencies:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="myuser",
        name="diffrhythm",
        tag="0.0.2",
        readme="Music generation with ASLP-lab/DiffRhythm")
    .from_base("parachutes/base-python:3.12.9")
    .set_user("root")
    .run_command("apt update && apt -y install espeak-ng")  # For text processing
    .set_user("chutes")
    .run_command("git clone https://github.com/ASLP-lab/DiffRhythm.git")
    .run_command("pip install -r DiffRhythm/requirements.txt")
    .run_command("pip install pybase64 py3langid")  # Additional dependencies
    .run_command("mv -f /app/DiffRhythm/* /app")  # Move to app directory
    .with_env("PYTHONPATH", "/app/infer")  # Set Python path
)
```

### Chute Configuration

Configure the service with appropriate GPU requirements:

```python
chute = Chute(
    username="myuser",
    name="diffrhythm-music",
    tagline="AI Music Generation with DiffRhythm",
    readme="Generate music from text descriptions and lyrics using advanced AI",
    image=image,
    node_selector=NodeSelector(gpu_count=1),  # Single GPU sufficient
)
```

### Model Initialization

Load and initialize all required models on startup:

```python
@chute.on_startup()
async def initialize(self):
    """
    Initialize DiffRhythm models and dependencies.
    """
    from huggingface_hub import snapshot_download
    import torchaudio
    import torch
    import soundfile
    from infer_utils import (
        decode_audio,
        get_lrc_token,
        get_negative_style_prompt,
        get_reference_latent,
        get_style_prompt,
        load_checkpoint,
        CNENTokenizer)
    from infer import inference
    from muq import MuQMuLan
    from model import DiT, CFM
    import json
    import os

    # Download required models
    revision = "613846abae8e5b869b3845a5dfabc9ecc37ecdab"
    repo_id = "ASLP-lab/DiffRhythm-full"
    path = snapshot_download(repo_id, revision=revision)

    vae_path = snapshot_download(
        "ASLP-lab/DiffRhythm-vae",
        revision="4656f626776f5f924c03471acb25bea6734e774f"
    )

    # Load model configuration
    dit_config_path = "/app/config/diffrhythm-1b.json"
    with open(dit_config_path) as f:
        model_config = json.load(f)

    # Initialize models
    dit_model_cls = DiT
    self.max_frames = 6144

    # CFM (Conditional Flow Matching) model
    self.cfm = CFM(
        transformer=dit_model_cls(**model_config["model"], max_frames=self.max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=self.max_frames
    ).to("cuda")

    # Load trained weights
    self.cfm = load_checkpoint(
        self.cfm,
        os.path.join(path, "cfm_model.pt"),
        device="cuda",
        use_ema=False
    )

    # Initialize tokenizer and style model
    self.tokenizer = CNENTokenizer()
    self.muq = MuQMuLan.from_pretrained(
        "OpenMuQ/MuQ-MuLan-large",
        revision="8a081dbcf84edd47ea7db3c4ecb8fd1ec1ddacfe"
    ).to("cuda")

    # Load VAE for audio decoding
    vae_ckpt_path = os.path.join(vae_path, "vae_model.pt")
    self.vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to("cuda")

    # Warmup with example generation
    await self._warmup_model()

    # Store utilities
    self.torchaudio = torchaudio
    self.torch = torch
    self.soundfile = soundfile
    self.decode_audio = decode_audio
    self.inference = inference
    self.get_lrc_token = get_lrc_token
    self.get_reference_latent = get_reference_latent
    self.get_style_prompt = get_style_prompt

async def _warmup_model(self):
    """Perform warmup generation to load models into memory."""
    from infer_utils import get_lrc_token, get_negative_style_prompt, get_reference_latent, get_style_prompt
    from infer import inference

    # Load example lyrics
    with open("/app/infer/example/eg_en_full.lrc", "r", encoding="utf-8") as infile:
        lrc = infile.read()

    # Prepare warmup data
    lrc_prompt, start_time = get_lrc_token(self.max_frames, lrc, self.tokenizer, "cuda")
    self.negative_style_prompt = get_negative_style_prompt("cuda")
    self.latent_prompt = get_reference_latent("cuda", self.max_frames)
    style_prompt = get_style_prompt(self.muq, prompt="classical genres, hopeful mood, piano.")

    # Perform warmup generation
    with self.torch.no_grad():
        generated_song = inference(
            cfm_model=self.cfm,
            vae_model=self.vae,
            cond=self.latent_prompt,
            text=lrc_prompt,
            duration=self.max_frames,
            style_prompt=style_prompt,
            negative_style_prompt=self.negative_style_prompt,
            start_time=start_time,
            chunked=True)

    # Save warmup output
    output_path = "/app/warmup.mp3"
    self.torchaudio.save(output_path, generated_song, sample_rate=44100, format="mp3")
```

### Audio Processing Utilities

Add utilities for handling audio input:

```python
import pybase64 as base64
import tempfile
from io import BytesIO
from loguru import logger

def load_audio(self, audio_b64):
    """
    Convert base64 audio to tensor for style extraction.
    """
    try:
        audio_bytes = BytesIO(base64.b64decode(audio_b64))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes.getvalue())
            temp_path = temp_file.name

        waveform, sample_rate = self.torchaudio.load(temp_path)
        return temp_path

    except Exception as exc:
        logger.error(f"Error loading audio: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input audio_b64 provided: {exc}")
```

### Lyrics Validation

Implement comprehensive lyrics validation with timing:

```python
def validate_lyrics(lyrics: str, total_length: int):
    """
    Validate LRC format lyrics for proper timing and format.
    """
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:05.2f}"

    previous_time = -1.0
    last_timestamp = 0.0

    try:
        for line_num, line in enumerate(lyrics.splitlines()):
            if not line.strip():
                continue

            # Check line length
            if len(line) > 256:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Line {line_num} exceeds 256 characters: {len(line)} chars")

            # Validate timestamp format
            valid_match = LRC_RE.match(line)
            if valid_match:
                minutes = int(valid_match.group(1))
                seconds = float(valid_match.group(2))
                current_time = minutes * 60 + seconds
                last_timestamp = max(last_timestamp, current_time)

                # Check chronological order
                if current_time < previous_time:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Line {line_num}: Timestamp {format_time(current_time)} "
                               f"is before previous timestamp {format_time(previous_time)}")
                previous_time = current_time

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error validating lyrics: {exc}")

    # Check total duration
    if last_timestamp > total_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Total duration ({format_time(last_timestamp)}) "
                   f"exceeds maximum allowed length ({format_time(total_length)})")
```

### Music Generation Endpoint

Create the main generation endpoint:

```python
import uuid
import os
from fastapi.responses import Response

@chute.cord(
    public_api_path="/generate",
    public_api_method="POST",
    stream=False,
    output_content_type="audio/mp3")
async def generate(self, args: InputArgs) -> Response:
    """
    Generate music from style prompts and/or lyrics.
    """
    input_path = None
    inference_kwargs = dict(
        cfm_model=self.cfm,
        vae_model=self.vae,
        cond=self.latent_prompt,
        duration=self.max_frames,
        negative_style_prompt=self.negative_style_prompt,
        chunked=True)

    # Extract style from prompt or reference audio
    style_prompt = None
    if args.style_prompt:
        style_prompt = self.get_style_prompt(self.muq, prompt=args.style_prompt)

    elif args.audio_b64:
        input_path = load_audio(self, args.audio_b64)
        try:
            style_prompt = self.get_style_prompt(self.muq, input_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input audio: {exc}")
        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)

    if style_prompt is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must provide either style_prompt or audio_b64!")

    inference_kwargs["style_prompt"] = style_prompt

    # Process lyrics if provided
    if args.lyrics:
        validate_lyrics(args.lyrics, 285)  # Max ~4.75 minutes

    lrc_prompt, start_time = self.get_lrc_token(
        self.max_frames, args.lyrics or "", self.tokenizer, "cuda"
    )
    inference_kwargs["text"] = lrc_prompt
    inference_kwargs["start_time"] = start_time

    # Generate the music
    output_path = f"/tmp/{uuid.uuid4()}.mp3"
    try:
        with self.torch.no_grad():
            generated_song = self.inference(**inference_kwargs)
            self.torchaudio.save(
                output_path, generated_song, sample_rate=44100, format="mp3"
            )

        # Return audio file
        with open(output_path, "rb") as infile:
            return Response(
                content=infile.read(),
                media_type="audio/mp3",
                headers={
                    "Content-Disposition": f"attachment; filename={uuid.uuid4()}.mp3",
                })

    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
```

## Advanced Features

### Style-Guided Generation

Create endpoint for style-specific music generation:

```python
class StyleRequest(BaseModel):
    style_description: str
    mood: Optional[str] = "neutral"
    genre: Optional[str] = "pop"
    instruments: Optional[str] = "piano, guitar"
    tempo: Optional[str] = "medium"

@chute.cord(public_api_path="/style_generate", method="POST")
async def generate_with_style(self, request: StyleRequest) -> Response:
    """Generate music with detailed style control."""

    # Construct detailed style prompt
    style_prompt = f"{request.genre} genre, {request.mood} mood, {request.instruments}"
    if request.tempo:
        style_prompt += f", {request.tempo} tempo"
    if request.style_description:
        style_prompt += f", {request.style_description}"

    # Generate using style prompt
    args = InputArgs(style_prompt=style_prompt)
    return await self.generate(args)
```

### Lyrics-to-Music with Timing

Example of properly formatted lyrics with timestamps:

```python
# Example LRC format lyrics
example_lyrics = """
[00:00.00]Verse 1
[00:05.50]In the morning light so bright
[00:10.00]I can see a better sight
[00:15.50]Dreams are calling out my name
[00:20.00]Nothing will be quite the same

[00:25.00]Chorus
[00:27.50]We are rising with the sun
[00:32.00]A new journey has begun
[00:37.50]Every step we take today
[00:42.00]Leads us down a brighter way

[00:47.00]Verse 2
[00:50.00]Through the valleys and the hills
[00:55.50]We will chase away our fears
[01:00.00]With the music in our hearts
[01:05.50]We will make a brand new start
"""

class LyricsRequest(BaseModel):
    lyrics: str
    style_prompt: str = "uplifting pop song, piano and strings"

@chute.cord(public_api_path="/lyrics_to_music", method="POST")
async def lyrics_to_music(self, request: LyricsRequest) -> Response:
    """Convert timestamped lyrics into a complete song."""

    args = InputArgs(
        style_prompt=request.style_prompt,
        lyrics=request.lyrics
    )
    return await self.generate(args)
```

### Reference Audio Style Transfer

Extract musical style from uploaded audio:

```python
class StyleTransferRequest(BaseModel):
    reference_audio_b64: str
    new_lyrics: Optional[str] = None
    style_blend: float = Field(default=1.0, ge=0.1, le=1.0)

@chute.cord(public_api_path="/style_transfer", method="POST")
async def style_transfer(self, request: StyleTransferRequest) -> Response:
    """Generate music using the style from reference audio."""

    args = InputArgs(
        audio_b64=request.reference_audio_b64,
        lyrics=request.new_lyrics
    )
    return await self.generate(args)
```

## Deployment and Usage

### Deploy the Service

```bash
# Build and deploy the music generation service
chutes deploy my_music_gen:chute

# Monitor the deployment
chutes chutes get my-music-gen
```

### Using the API

#### Generate with Style Prompt

```bash
curl -X POST "https://myuser-my-music-gen.chutes.ai/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "style_prompt": "upbeat electronic dance music, synthesizers, energetic"
  }' \
  --output generated_music.mp3
```

#### Generate with Lyrics

```bash
curl -X POST "https://myuser-my-music-gen.chutes.ai/lyrics_to_music" \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[00:00.00]Hello world\n[00:05.00]This is my song\n[00:10.00]Made with AI",
    "style_prompt": "acoustic folk, guitar and violin, heartfelt"
  }' \
  --output lyrical_song.mp3
```

#### Python Client Example

```python
import requests
import base64

class MusicGenerator:
    def __init__(self, base_url):
        self.base_url = base_url

    def generate_from_style(self, style_prompt):
        """Generate music from style description."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={"style_prompt": style_prompt}
        )

        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Generation failed: {response.status_code}")

    def generate_from_lyrics(self, lyrics, style="pop"):
        """Generate music from timestamped lyrics."""
        response = requests.post(
            f"{self.base_url}/lyrics_to_music",
            json={
                "lyrics": lyrics,
                "style_prompt": f"{style} style, full band arrangement"
            }
        )
        return response.content

    def style_transfer(self, reference_audio_path, new_lyrics=None):
        """Generate music using style from reference audio."""
        with open(reference_audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = {"reference_audio_b64": audio_b64}
        if new_lyrics:
            payload["new_lyrics"] = new_lyrics

        response = requests.post(
            f"{self.base_url}/style_transfer",
            json=payload
        )
        return response.content

# Usage example
generator = MusicGenerator("https://myuser-my-music-gen.chutes.ai")

# Generate upbeat electronic music
music = generator.generate_from_style(
    "energetic electronic dance music, heavy bass, futuristic sounds"
)

with open("edm_track.mp3", "wb") as f:
    f.write(music)

# Generate from lyrics
lyrics = """
[00:00.00]Verse 1
[00:03.00]AI creates the beat
[00:06.00]Technology so sweet
[00:09.00]Music from the future
[00:12.00]Is here to greet ya
"""

song = generator.generate_from_lyrics(lyrics, "electronic pop")
with open("ai_song.mp3", "wb") as f:
    f.write(song)
```

## Best Practices

### 1. Lyrics Formatting

```python
# Good LRC format - clear timing and structure
good_lyrics = """
[00:00.00]Intro
[00:08.00]Verse 1
[00:10.50]Walking down the street tonight
[00:15.00]City lights are shining bright
[00:20.50]Every step I take feels right
[00:25.00]In this neon-colored light

[00:30.00]Chorus
[00:32.50]We are alive, we are free
[00:37.00]This is who we're meant to be
[00:42.50]Dancing through eternity
[00:47.00]In perfect harmony
"""

# Bad format - inconsistent timing
bad_lyrics = """
[00:00]Start
[0:5]Some lyrics here
[15.5]More lyrics without proper format
Random text without timestamp
"""
```

### 2. Style Prompt Engineering

```python
# Effective style prompts are specific and descriptive
effective_styles = [
    "jazz ballad, piano and saxophone, slow tempo, romantic mood",
    "rock anthem, electric guitars, powerful drums, energetic",
    "classical orchestral, strings and brass, dramatic, cinematic",
    "ambient electronic, synthesizers, dreamy, ethereal atmosphere",
    "country folk, acoustic guitar, harmonica, storytelling style"
]

# Avoid vague prompts
vague_styles = [
    "good music",
    "nice song",
    "popular style"
]
```

### 3. Audio Quality Optimization

```python
# For highest quality output
@chute.cord(public_api_path="/hq_generate", method="POST")
async def high_quality_generate(self, args: InputArgs) -> Response:
    """Generate high-quality music with extended processing."""

    # Use maximum duration for better quality
    inference_kwargs = dict(
        cfm_model=self.cfm,
        vae_model=self.vae,
        cond=self.latent_prompt,
        duration=self.max_frames,  # Use full duration
        negative_style_prompt=self.negative_style_prompt,
        chunked=False,  # Don't chunk for better coherence
    )

    # ... rest of generation logic
```

### 4. Error Handling and Validation

```python
def validate_audio_input(audio_b64: str, max_size_mb: int = 10):
    """Validate audio input size and format."""
    try:
        audio_data = base64.b64decode(audio_b64)
        size_mb = len(audio_data) / (1024 * 1024)

        if size_mb > max_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
            )

        return audio_data
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio data: {str(e)}"
        )
```

## Performance and Scaling

### Memory Optimization

```python
# Clear GPU memory between generations
@chute.cord(public_api_path="/generate", method="POST")
async def generate_optimized(self, args: InputArgs) -> Response:
    """Memory-optimized generation."""

    try:
        # Clear cache before generation
        if hasattr(self, 'torch'):
            self.torch.cuda.empty_cache()

        # Generate music
        result = await self.generate(args)

        return result

    finally:
        # Clean up after generation
        if hasattr(self, 'torch'):
            self.torch.cuda.empty_cache()
```

### Concurrent Processing

```python
# Configure for multiple concurrent generations
chute = Chute(
    username="myuser",
    name="diffrhythm-music",
    image=image,
    node_selector=NodeSelector(
        gpu_count=2,  # Multiple GPUs for parallel processing
        min_vram_gb_per_gpu=24
    ),
    concurrency=4,  # Handle multiple requests
)
```

## Monitoring and Troubleshooting

### Common Issues and Solutions

```bash
# Check service health
chutes chutes get my-music-gen

# View generation logs
chutes chutes logs my-music-gen --tail 50

# Monitor GPU utilization
chutes chutes metrics my-music-gen
```

### Performance Monitoring

```python
import time
from loguru import logger

@chute.cord(public_api_path="/generate_timed", method="POST")
async def generate_with_timing(self, args: InputArgs) -> Response:
    """Generation with performance monitoring."""

    start_time = time.time()

    try:
        result = await self.generate(args)

        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")

        return result

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Generation failed after {error_time:.2f} seconds: {e}")
        raise
```

## Next Steps

- **Custom Models**: Train DiffRhythm on your own musical datasets
- **Style Control**: Experiment with different musical genres and moods
- **Integration**: Build music creation apps and platforms
- **Real-time**: Implement streaming music generation

For more advanced examples, see:

- [Audio Processing](/docs/examples/audio-processing)
- [Custom Training](/docs/examples/custom-training)
- [Real-time Streaming](/docs/examples/streaming-responses)
