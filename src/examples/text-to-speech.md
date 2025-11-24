# Text-to-Speech with CSM-1B

This guide demonstrates how to build a sophisticated text-to-speech (TTS) service using CSM-1B (Conversational Speech Model), capable of generating natural-sounding speech with context awareness and multiple speaker support.

## Overview

CSM-1B from Sesame is a state-of-the-art speech generation model that:

- Generates high-quality speech from text input
- Supports multiple speakers (2 speakers available)
- Uses context from previous audio/text for continuity
- Employs Llama backbone with specialized audio decoder
- Produces Mimi audio codes for natural speech output
- Supports configurable duration limits

## Complete Implementation

### Input Schema Design

Define comprehensive input validation for TTS generation:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Context(BaseModel):
    text: str
    speaker: int = Field(0, gte=0, lte=1)
    audio_b64: str  # Base64 encoded reference audio

class InputArgs(BaseModel):
    text: str
    context: Optional[List[Context]] = []
    speaker: Optional[int] = Field(1, gte=0, lte=1)
    max_duration_ms: Optional[int] = 10000  # Maximum output duration
```

### Custom Image with CSM-1B

Build a custom image with all required dependencies:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="myuser",
        name="csm-1b",
        tag="0.0.2",
        readme="## Text-to-speech using sesame/csm-1b")
    .from_base("parachutes/base-python:3.12.9")
    .run_command(
        "pip install -r https://huggingface.co/chutesai/csm-1b/resolve/main/requirements.txt"
    )
    .run_command("pip install pybase64")  # For audio encoding/decoding
    .run_command(
        "wget -O /app/generator.py https://huggingface.co/chutesai/csm-1b/resolve/main/generator.py"
    )
    .run_command(
        "wget -O /app/models.py https://huggingface.co/chutesai/csm-1b/resolve/main/models.py"
    )
    .run_command(
        "wget -O /app/watermarking.py https://huggingface.co/chutesai/csm-1b/resolve/main/watermarking.py"
    )
)
```

### Chute Configuration

Configure the service with appropriate GPU requirements:

```python
chute = Chute(
    username="myuser",
    name="csm-1b-tts",
    tagline="High-quality text-to-speech with CSM-1B",
    readme="CSM (Conversational Speech Model) generates natural speech from text with context awareness and multiple speaker support.",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=24  # 24GB required for optimal performance
    ))
```

### Model Initialization

Load and initialize the CSM-1B model on startup:

```python
@chute.on_startup()
async def initialize(self):
    """
    Initialize the CSM-1B model and perform warmup.
    """
    from huggingface_hub import snapshot_download
    from generator import Generator
    from models import Model
    import torchaudio
    import torch

    # Download the model with specific revision
    revision = "01e2ed64be01915391ec7881f666d6dda0e1d509"
    snapshot_download("chutesai/csm-1b", revision=revision)

    # Store torchaudio for later use
    self.torchaudio = torchaudio

    # Initialize the model
    model = Model.from_pretrained("chutesai/csm-1b", revision=revision)
    model.to(device="cuda", dtype=torch.bfloat16)

    # Create the generator
    self.generator = Generator(model)

    # Warmup generation to load models into memory
    _ = self.generator.generate(
        text="Warming up Sesame...",
        speaker=0,
        context=[],
        max_audio_length_ms=10000)
```

### Audio Processing Utilities

Add utilities for handling audio input and output:

```python
import pybase64 as base64
import tempfile
import os
from io import BytesIO
from loguru import logger
from fastapi import HTTPException, status

def load_audio(self, audio_b64):
    """
    Convert base64 audio data into audio tensor.
    Ensures the output is a 1D tensor [T] for compatibility.
    """
    try:
        # Decode base64 to audio bytes
        audio_bytes = BytesIO(base64.b64decode(audio_b64))

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes.getvalue())
            temp_path = temp_file.name

        # Load audio with torchaudio
        waveform, sample_rate = self.torchaudio.load(temp_path)
        os.unlink(temp_path)  # Clean up temp file

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Resample to model's expected sample rate
        audio_tensor = self.torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=self.generator.sample_rate)

        # Ensure 1D tensor
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        return audio_tensor

    except Exception as exc:
        logger.error(f"Error loading audio: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input audio_b64 provided: {exc}")
```

### Text-to-Speech Endpoint

Create the main TTS generation endpoint:

```python
import uuid
from fastapi import Response

@chute.cord(
    public_api_path="/speak",
    public_api_method="POST",
    stream=False,
    output_content_type="audio/wav")
async def speak(self, args: InputArgs) -> Response:
    """
    Convert text to speech with optional context.
    """
    from generator import Segment

    # Process context if provided
    segments = []
    if args.context:
        for ctx in args.context:
            audio_tensor = load_audio(self, ctx.audio_b64)
            segments.append(
                Segment(
                    text=ctx.text,
                    speaker=ctx.speaker,
                    audio=audio_tensor)
            )

    # Generate speech audio
    audio = self.generator.generate(
        text=args.text,
        speaker=args.speaker,
        context=segments,
        max_audio_length_ms=args.max_duration_ms)

    # Save to temporary file
    path = f"/tmp/{uuid.uuid4()}.wav"
    self.torchaudio.save(
        path,
        audio.unsqueeze(0).cpu(),
        self.generator.sample_rate
    )

    try:
        # Return audio file
        with open(path, "rb") as infile:
            return Response(
                content=infile.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={uuid.uuid4()}.wav",
                })
    finally:
        # Clean up temporary file
        if os.path.exists(path):
            os.remove(path)
```

## Advanced Features

### Multi-Speaker Conversation

Create endpoint for generating conversation between speakers:

```python
class ConversationTurn(BaseModel):
    speaker: int = Field(ge=0, le=1)
    text: str
    pause_ms: Optional[int] = Field(default=500, ge=0, le=2000)

class ConversationInput(BaseModel):
    turns: List[ConversationTurn]
    max_total_duration_ms: int = Field(default=30000, ge=5000, le=60000)

@chute.cord(public_api_path="/conversation", method="POST")
async def generate_conversation(self, args: ConversationInput) -> Response:
    """Generate a conversation between multiple speakers."""
    from generator import Segment

    conversation_audio = []
    context_segments = []

    for turn in args.turns:
        # Generate speech for this turn with accumulated context
        audio = self.generator.generate(
            text=turn.text,
            speaker=turn.speaker,
            context=context_segments,
            max_audio_length_ms=args.max_total_duration_ms // len(args.turns))

        conversation_audio.append(audio)

        # Add silence between turns
        if turn.pause_ms > 0:
            silence_samples = int(turn.pause_ms * self.generator.sample_rate / 1000)
            silence = torch.zeros(silence_samples)
            conversation_audio.append(silence)

        # Add this turn to context for future turns
        context_segments.append(
            Segment(
                text=turn.text,
                speaker=turn.speaker,
                audio=audio)
        )

    # Concatenate all audio
    full_audio = torch.cat(conversation_audio, dim=0)

    # Save and return
    path = f"/tmp/conversation_{uuid.uuid4()}.wav"
    self.torchaudio.save(path, full_audio.unsqueeze(0).cpu(), self.generator.sample_rate)

    try:
        with open(path, "rb") as infile:
            return Response(
                content=infile.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=conversation.wav"})
    finally:
        if os.path.exists(path):
            os.remove(path)
```

### Voice Cloning with Reference Audio

Clone a voice from a reference audio sample:

```python
class VoiceCloningInput(BaseModel):
    text: str
    reference_audio_b64: str
    reference_text: str  # Text that was spoken in reference audio
    max_duration_ms: int = Field(default=15000, ge=1000, le=30000)

@chute.cord(public_api_path="/clone_voice", method="POST")
async def clone_voice(self, args: VoiceCloningInput) -> Response:
    """Generate speech using a reference voice sample."""
    from generator import Segment

    # Load reference audio
    reference_audio = load_audio(self, args.reference_audio_b64)

    # Create context segment from reference
    reference_segment = Segment(
        text=args.reference_text,
        speaker=0,  # Use speaker 0 as base
        audio=reference_audio)

    # Generate new speech with reference voice characteristics
    audio = self.generator.generate(
        text=args.text,
        speaker=0,
        context=[reference_segment],
        max_audio_length_ms=args.max_duration_ms)

    # Save and return
    path = f"/tmp/cloned_{uuid.uuid4()}.wav"
    self.torchaudio.save(path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

    try:
        with open(path, "rb") as infile:
            return Response(
                content=infile.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=cloned_voice.wav"})
    finally:
        if os.path.exists(path):
            os.remove(path)
```

### Batch Processing

Process multiple texts efficiently:

```python
class BatchTTSInput(BaseModel):
    texts: List[str] = Field(max_items=10)  # Limit batch size
    speaker: int = Field(default=0, ge=0, le=1)
    max_duration_per_text_ms: int = Field(default=10000, ge=1000, le=20000)

@chute.cord(public_api_path="/batch_speak", method="POST")
async def batch_speak(self, args: BatchTTSInput) -> List[str]:
    """Generate speech for multiple texts and return as base64 list."""
    results = []

    for text in args.texts:
        # Generate audio for each text
        audio = self.generator.generate(
            text=text,
            speaker=args.speaker,
            context=[],
            max_audio_length_ms=args.max_duration_per_text_ms)

        # Convert to WAV bytes
        path = f"/tmp/batch_{uuid.uuid4()}.wav"
        self.torchaudio.save(path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

        try:
            with open(path, "rb") as infile:
                audio_b64 = base64.b64encode(infile.read()).decode()
                results.append(audio_b64)
        finally:
            if os.path.exists(path):
                os.remove(path)

    return results
```

## Deployment and Usage

### Deploy the Service

```bash
# Build and deploy the TTS service
chutes deploy my_tts:chute

# Monitor the deployment
chutes chutes get my-tts
```

### Using the API

#### Basic Text-to-Speech

```bash
curl -X POST "https://myuser-my-tts.chutes.ai/speak" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a demonstration of high-quality text-to-speech synthesis.",
    "speaker": 0,
    "max_duration_ms": 15000
  }' \
  --output speech.wav
```

#### Voice Cloning

```bash
# First, encode your reference audio to base64
# base64 -i reference.wav > reference.b64

curl -X POST "https://myuser-my-tts.chutes.ai/clone_voice" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is new text spoken in the reference voice",
    "reference_audio_b64": "'$(cat reference.b64)'",
    "reference_text": "Original text that was spoken in the reference audio",
    "max_duration_ms": 20000
  }' \
  --output cloned_speech.wav
```

#### Python Client Example

```python
import requests
import base64
import io
from pydantic import BaseModel
from typing import List, Optional

class TTSClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def speak(self, text: str, speaker: int = 0, max_duration_ms: int = 10000) -> bytes:
        """Generate speech from text."""
        response = requests.post(
            f"{self.base_url}/speak",
            json={
                "text": text,
                "speaker": speaker,
                "max_duration_ms": max_duration_ms
            }
        )

        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"TTS failed: {response.status_code} - {response.text}")

    def clone_voice(self, text: str, reference_audio_path: str, reference_text: str) -> bytes:
        """Generate speech using voice cloning."""
        # Encode reference audio
        with open(reference_audio_path, "rb") as f:
            reference_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            f"{self.base_url}/clone_voice",
            json={
                "text": text,
                "reference_audio_b64": reference_b64,
                "reference_text": reference_text,
                "max_duration_ms": 20000
            }
        )

        return response.content

    def generate_conversation(self, turns: List[dict]) -> bytes:
        """Generate a conversation between speakers."""
        response = requests.post(
            f"{self.base_url}/conversation",
            json={
                "turns": turns,
                "max_total_duration_ms": 30000
            }
        )

        return response.content

    def batch_speak(self, texts: List[str], speaker: int = 0) -> List[bytes]:
        """Generate speech for multiple texts."""
        response = requests.post(
            f"{self.base_url}/batch_speak",
            json={
                "texts": texts,
                "speaker": speaker,
                "max_duration_per_text_ms": 10000
            }
        )

        if response.status_code == 200:
            b64_results = response.json()
            return [base64.b64decode(b64) for b64 in b64_results]
        else:
            raise Exception(f"Batch TTS failed: {response.status_code}")

# Usage examples
client = TTSClient("https://myuser-my-tts.chutes.ai")

# Basic TTS
speech_audio = client.speak("Hello, world! This is synthesized speech.")
with open("hello.wav", "wb") as f:
    f.write(speech_audio)

# Voice cloning
cloned_audio = client.clone_voice(
    text="This is new content in the cloned voice",
    reference_audio_path="reference_voice.wav",
    reference_text="This was the original reference text"
)
with open("cloned.wav", "wb") as f:
    f.write(cloned_audio)

# Conversation generation
conversation_turns = [
    {"speaker": 0, "text": "Hello, how are you today?", "pause_ms": 1000},
    {"speaker": 1, "text": "I'm doing great, thanks for asking!", "pause_ms": 800},
    {"speaker": 0, "text": "That's wonderful to hear.", "pause_ms": 500}
]

conversation_audio = client.generate_conversation(conversation_turns)
with open("conversation.wav", "wb") as f:
    f.write(conversation_audio)
```

## Best Practices

### 1. Text Preprocessing

```python
import re

def preprocess_text(text: str) -> str:
    """Clean and prepare text for TTS."""
    # Expand common abbreviations
    text = text.replace("Dr.", "Doctor")
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Missus")
    text = text.replace("&", "and")

    # Handle numbers (basic example)
    text = re.sub(r'\b(\d+)\b', lambda m: num_to_words(int(m.group(1))), text)

    # Remove excessive punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)

    return text.strip()

def num_to_words(num: int) -> str:
    """Convert numbers to words (basic implementation)."""
    if num == 0:
        return "zero"
    elif num == 1:
        return "one"
    # Add more number conversions as needed
    else:
        return str(num)  # Fallback
```

### 2. Context Management

```python
class ContextManager:
    """Manage conversation context for better continuity."""

    def __init__(self, max_context_length: int = 5):
        self.context_segments = []
        self.max_length = max_context_length

    def add_segment(self, text: str, speaker: int, audio_tensor):
        """Add a new segment to context."""
        from generator import Segment

        segment = Segment(text=text, speaker=speaker, audio=audio_tensor)
        self.context_segments.append(segment)

        # Keep only recent context
        if len(self.context_segments) > self.max_length:
            self.context_segments = self.context_segments[-self.max_length:]

    def get_context(self) -> List:
        """Get current context for generation."""
        return self.context_segments.copy()

    def clear(self):
        """Clear all context."""
        self.context_segments = []

# Usage in endpoint
@chute.cord(public_api_path="/contextual_speak", method="POST")
async def contextual_speak(self, args: InputArgs) -> Response:
    """Generate speech with persistent context."""
    if not hasattr(self, 'context_manager'):
        self.context_manager = ContextManager()

    # Generate with context
    audio = self.generator.generate(
        text=args.text,
        speaker=args.speaker,
        context=self.context_manager.get_context(),
        max_audio_length_ms=args.max_duration_ms)

    # Add to context for future generations
    self.context_manager.add_segment(args.text, args.speaker, audio)

    # Return audio...
```

### 3. Quality Control

```python
def validate_audio_quality(audio_tensor, sample_rate: int) -> bool:
    """Check generated audio quality."""
    import torch

    # Check for silence (all zeros)
    if torch.all(audio_tensor == 0):
        return False

    # Check for clipping
    if torch.max(torch.abs(audio_tensor)) > 0.99:
        return False

    # Check minimum duration (avoid too short clips)
    min_duration_ms = 500
    min_samples = int(min_duration_ms * sample_rate / 1000)
    if len(audio_tensor) < min_samples:
        return False

    return True

@chute.cord(public_api_path="/quality_speak", method="POST")
async def quality_controlled_speak(self, args: InputArgs) -> Response:
    """Generate speech with quality validation."""
    max_retries = 3

    for attempt in range(max_retries):
        audio = self.generator.generate(
            text=args.text,
            speaker=args.speaker,
            context=[],
            max_audio_length_ms=args.max_duration_ms)

        if validate_audio_quality(audio, self.generator.sample_rate):
            # Quality passed, return audio
            break
        else:
            logger.warning(f"Audio quality check failed, attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate quality audio after multiple attempts"
                )

    # Save and return validated audio...
```

## Performance Optimization

### Memory Management

```python
@chute.cord(public_api_path="/optimized_speak", method="POST")
async def optimized_speak(self, args: InputArgs) -> Response:
    """Memory-optimized speech generation."""
    import torch

    try:
        # Clear cache before generation
        torch.cuda.empty_cache()

        # Generate with memory efficiency
        with torch.inference_mode():
            audio = self.generator.generate(
                text=args.text,
                speaker=args.speaker,
                context=args.context,
                max_audio_length_ms=args.max_duration_ms)

        # Process and return immediately
        path = f"/tmp/{uuid.uuid4()}.wav"
        self.torchaudio.save(path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

        # Read and clean up immediately
        with open(path, "rb") as infile:
            content = infile.read()
        os.remove(path)

        return Response(
            content=content,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=speech.wav"})

    finally:
        # Always clear cache after generation
        torch.cuda.empty_cache()
```

### Caching for Repeated Requests

```python
import hashlib
from typing import Dict

class TTSCache:
    """Simple cache for TTS results."""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, bytes] = {}
        self.max_size = max_size

    def get_key(self, text: str, speaker: int) -> str:
        """Generate cache key."""
        content = f"{text}_{speaker}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, speaker: int) -> Optional[bytes]:
        """Get cached result."""
        key = self.get_key(text, speaker)
        return self.cache.get(key)

    def set(self, text: str, speaker: int, audio_bytes: bytes):
        """Cache result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        key = self.get_key(text, speaker)
        self.cache[key] = audio_bytes

# Add to chute initialization
@chute.on_startup()
async def initialize_with_cache(self):
    # ... existing initialization ...
    self.tts_cache = TTSCache(max_size=200)

@chute.cord(public_api_path="/cached_speak", method="POST")
async def cached_speak(self, args: InputArgs) -> Response:
    """TTS with caching for repeated requests."""

    # Check cache first (only for simple requests without context)
    if not args.context:
        cached_result = self.tts_cache.get(args.text, args.speaker)
        if cached_result:
            return Response(
                content=cached_result,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=cached_speech.wav"})

    # Generate new audio
    audio = self.generator.generate(
        text=args.text,
        speaker=args.speaker,
        context=[],
        max_audio_length_ms=args.max_duration_ms)

    # Save to file and cache
    path = f"/tmp/{uuid.uuid4()}.wav"
    self.torchaudio.save(path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

    with open(path, "rb") as infile:
        audio_bytes = infile.read()
    os.remove(path)

    # Cache result
    if not args.context:
        self.tts_cache.set(args.text, args.speaker, audio_bytes)

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"})
```

## Monitoring and Troubleshooting

### Performance Monitoring

```bash
# Check service health
chutes chutes get my-tts

# View generation logs
chutes chutes logs my-tts --tail 100

# Monitor GPU utilization
chutes chutes metrics my-tts
```

### Common Issues and Solutions

```python
# Handle common TTS issues
@chute.cord(public_api_path="/robust_speak", method="POST")
async def robust_speak(self, args: InputArgs) -> Response:
    """TTS with comprehensive error handling."""

    try:
        # Preprocess text
        processed_text = preprocess_text(args.text)

        # Validate text length
        if len(processed_text) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Text too long. Maximum 1000 characters allowed."
            )

        # Generate audio
        audio = self.generator.generate(
            text=processed_text,
            speaker=args.speaker,
            context=[],
            max_audio_length_ms=args.max_duration_ms)

        # Validate output
        if not validate_audio_quality(audio, self.generator.sample_rate):
            raise HTTPException(
                status_code=500,
                detail="Generated audio failed quality checks"
            )

        # Return successful result
        path = f"/tmp/{uuid.uuid4()}.wav"
        self.torchaudio.save(path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

        with open(path, "rb") as infile:
            content = infile.read()
        os.remove(path)

        return Response(
            content=content,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"})

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=503,
            detail="GPU memory exhausted. Please try again or reduce duration."
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(e)}"
        )
```

## Next Steps

- **Custom Voice Training**: Train CSM-1B on your own voice data
- **Multilingual Support**: Experiment with different languages
- **Real-time Streaming**: Implement streaming TTS for live applications
- **Integration**: Build voice assistants and interactive applications

For more advanced examples, see:

- [Real-time Streaming](/docs/examples/streaming-responses)
- [Custom Training](/docs/examples/custom-training)
- [Audio Processing](/docs/examples/audio-processing)
