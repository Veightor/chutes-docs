# Audio Processing with Chutes

This guide demonstrates comprehensive audio processing capabilities using Chutes, from basic audio manipulation to advanced machine learning tasks like speech recognition, synthesis, and audio analysis.

## Overview

Audio processing with Chutes enables:

- **Speech Recognition**: Convert speech to text with high accuracy
- **Text-to-Speech**: Generate natural-sounding speech from text
- **Audio Enhancement**: Noise reduction, audio restoration, and quality improvement
- **Music Analysis**: Beat detection, genre classification, and audio fingerprinting
- **Real-time Processing**: Stream audio processing with low latency
- **Multi-format Support**: Handle various audio formats (WAV, MP3, FLAC, etc.)

## Quick Start

### Basic Audio Processing Setup

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64

class AudioProcessingConfig(BaseModel):
    input_format: str = "wav"
    output_format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16

# Audio processing image with all dependencies
audio_image = (
    Image(
        username="myuser",
        name="audio-processing",
        tag="1.0.0",
        python_version="3.11"
    )
    .run_command("""
        apt-get update && apt-get install -y \\
        ffmpeg \\
        libsndfile1 \\
        libsndfile1-dev \\
        portaudio19-dev \\
        libportaudio2 \\
        libportaudiocpp0 \\
        pulseaudio
    """)
    .run_command("pip install librosa==0.10.1 soundfile==0.12.1 pydub==0.25.1 pyaudio==0.2.11 numpy==1.24.3 scipy==1.11.4 torch==2.1.0 torchaudio==2.1.0 transformers==4.35.0 whisper==1.1.10")
    .add("./audio_utils", "/app/audio_utils")
    .add("./models", "/app/models")
)
```

## Speech Recognition

### Whisper-based Speech-to-Text

```python
import whisper
import librosa
import soundfile as sf
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tempfile
import os

class TranscriptionRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None
    task: str = "transcribe"  # "transcribe" or "translate"
    temperature: float = 0.0
    word_timestamps: bool = False

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    segments: List[Dict[str, Any]]
    processing_time_ms: float

class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
        self.model_size = model_size

    def preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio for Whisper"""
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            # Load and resample to 16kHz (Whisper requirement)
            audio, sr = librosa.load(temp_path, sr=16000, mono=True)
            return audio
        finally:
            os.unlink(temp_path)

    def transcribe_audio(self, audio_data: bytes, options: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe audio using Whisper"""
        import time
        start_time = time.time()

        # Preprocess audio
        audio = self.preprocess_audio(audio_data)

        # Transcription options
        transcribe_options = {
            "language": options.language,
            "task": options.task,
            "temperature": options.temperature,
            "word_timestamps": options.word_timestamps
        }

        # Remove None values
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}

        # Transcribe
        result = self.model.transcribe(audio, **transcribe_options)

        processing_time = (time.time() - start_time) * 1000

        return TranscriptionResponse(
            text=result["text"].strip(),
            language=result["language"],
            segments=result["segments"],
            processing_time_ms=processing_time
        )

# Global transcriber instance
transcriber = None

def initialize_transcriber(model_size: str = "base"):
    """Initialize Whisper transcriber"""
    global transcriber
    transcriber = WhisperTranscriber(model_size)
    return {"status": "initialized", "model": model_size}

async def transcribe_speech(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Speech recognition endpoint"""
    request = TranscriptionRequest(**inputs)

    # Decode base64 audio
    audio_data = base64.b64decode(request.audio_base64)

    # Transcribe
    result = transcriber.transcribe_audio(audio_data, request)

    return result.dict()
```

### Real-time Speech Recognition

```python
import pyaudio
import threading
import queue
import numpy as np
from collections import deque

class RealTimeTranscriber:
    def __init__(self, model_size: str = "base", chunk_duration: float = 2.0):
        self.model = whisper.load_model(model_size)
        self.chunk_duration = chunk_duration
        self.sample_rate = 16000
        self.chunk_size = int(chunk_duration * self.sample_rate)

        # Audio streaming setup
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_buffer = deque(maxlen=self.sample_rate * 10)  # 10 second buffer

    def start_recording(self):
        """Start real-time audio recording"""
        self.is_recording = True

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )

        stream.start_stream()

        # Start transcription thread
        transcription_thread = threading.Thread(target=self._transcription_worker)
        transcription_thread.start()

        return stream, audio

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)

        # Check if we have enough data for a chunk
        if len(self.audio_buffer) >= self.chunk_size:
            chunk = np.array(list(self.audio_buffer)[-self.chunk_size:])
            self.audio_queue.put(chunk)

        return (None, pyaudio.paContinue)

    def _transcription_worker(self):
        """Background transcription worker"""
        while self.is_recording:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Transcribe chunk
                result = self.model.transcribe(audio_chunk, language="en")

                if result["text"].strip():
                    yield {
                        "text": result["text"].strip(),
                        "timestamp": time.time(),
                        "confidence": self._estimate_confidence(result)
                    }

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")

    def _estimate_confidence(self, result):
        """Estimate transcription confidence"""
        # Simple confidence estimation based on segment probabilities
        if "segments" in result and result["segments"]:
            avg_prob = np.mean([seg.get("avg_logprob", -1.0) for seg in result["segments"]])
            return max(0.0, min(1.0, (avg_prob + 1.0)))
        return 0.5
```

## Text-to-Speech

### Advanced TTS with Coqui TTS

```python
import torch
from TTS.api import TTS
import tempfile
import base64
from typing import Optional

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: str = "en"
    speed: float = 1.0
    emotion: Optional[str] = None

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    processing_time_ms: float

class AdvancedTTSService:
    def __init__(self):
        # Initialize Coqui TTS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load multi-speaker TTS model
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(self.device)

        # Available speakers and languages
        self.speakers = self.tts.speakers if hasattr(self.tts, 'speakers') else []
        self.languages = self.tts.languages if hasattr(self.tts, 'languages') else ["en"]

    def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech from text"""
        import time
        start_time = time.time()

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_path = temp_file.name

        try:
            # Synthesize speech
            self.tts.tts_to_file(
                text=request.text,
                file_path=output_path,
                speaker=request.speaker,
                language=request.language,
                speed=request.speed
            )

            # Load generated audio
            audio, sample_rate = librosa.load(output_path, sr=None)

            # Apply speed adjustment if needed
            if request.speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=request.speed)

            # Convert to base64
            with open(output_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode()

            processing_time = (time.time() - start_time) * 1000
            duration = len(audio) / sample_rate

            return TTSResponse(
                audio_base64=audio_base64,
                sample_rate=sample_rate,
                duration_seconds=duration,
                processing_time_ms=processing_time
            )

        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)

# Global TTS service
tts_service = None

def initialize_tts():
    """Initialize TTS service"""
    global tts_service
    tts_service = AdvancedTTSService()
    return {
        "status": "initialized",
        "speakers": tts_service.speakers,
        "languages": tts_service.languages
    }

async def synthesize_text(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Text-to-speech endpoint"""
    request = TTSRequest(**inputs)
    result = tts_service.synthesize_speech(request)
    return result.dict()
```

## Audio Enhancement

### Noise Reduction and Audio Restoration

```python
import librosa
import numpy as np
from scipy import signal
import noisereduce as nr

class AudioEnhancer:
    def __init__(self):
        self.sample_rate = 22050

    def reduce_noise(self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Reduce background noise using spectral subtraction"""
        if noise_profile is None:
            # Use first 0.5 seconds as noise profile
            noise_duration = int(0.5 * self.sample_rate)
            noise_profile = audio[:noise_duration]

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=0.8
        )

        return reduced_noise

    def normalize_audio(self, audio: np.ndarray, target_level: float = -23.0) -> np.ndarray:
        """Normalize audio to target loudness level (LUFS)"""
        # Simple peak normalization
        current_peak = np.max(np.abs(audio))
        if current_peak > 0:
            target_peak = 10 ** (target_level / 20)
            normalization_factor = target_peak / current_peak
            return audio * normalization_factor
        return audio

    def apply_eq(self, audio: np.ndarray, eq_bands: List[Dict[str, float]]) -> np.ndarray:
        """Apply parametric EQ with multiple bands"""
        processed_audio = audio.copy()

        for band in eq_bands:
            frequency = band["frequency"]
            gain = band["gain"]
            q_factor = band.get("q", 1.0)

            # Design filter
            nyquist = self.sample_rate / 2
            normalized_freq = frequency / nyquist

            if gain != 0:
                # Peaking filter
                b, a = signal.iirpeak(normalized_freq, Q=q_factor)
                if gain > 0:
                    # Boost
                    boost_factor = 10 ** (gain / 20)
                    processed_audio = signal.lfilter(b * boost_factor, a, processed_audio)
                else:
                    # Cut
                    cut_factor = 10 ** (-abs(gain) / 20)
                    processed_audio = signal.lfilter(b * cut_factor, a, processed_audio)

        return processed_audio

    def remove_clicks_pops(self, audio: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Remove clicks and pops from audio"""
        # Detect sudden amplitude changes
        diff = np.diff(audio)
        click_indices = np.where(np.abs(diff) > threshold)[0]

        # Interpolate over detected clicks
        for idx in click_indices:
            if idx > 0 and idx < len(audio) - 1:
                # Linear interpolation
                audio[idx] = (audio[idx-1] + audio[idx+1]) / 2

        return audio

async def enhance_audio(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Audio enhancement endpoint"""
    # Decode input audio
    audio_base64 = inputs["audio_base64"]
    audio_data = base64.b64decode(audio_base64)

    # Load audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file.write(audio_data)
        temp_file.flush()
        audio, sr = librosa.load(temp_file.name, sr=None)

    enhancer = AudioEnhancer()

    # Apply enhancements based on options
    options = inputs.get("options", {})

    if options.get("reduce_noise", False):
        audio = enhancer.reduce_noise(audio)

    if options.get("normalize", False):
        target_level = options.get("target_level", -23.0)
        audio = enhancer.normalize_audio(audio, target_level)

    if "eq_bands" in options:
        audio = enhancer.apply_eq(audio, options["eq_bands"])

    if options.get("remove_clicks", False):
        audio = enhancer.remove_clicks_pops(audio)

    # Save enhanced audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        sf.write(temp_file.name, audio, sr)
        temp_file.seek(0)
        enhanced_audio_base64 = base64.b64encode(temp_file.read()).decode()

    return {
        "enhanced_audio_base64": enhanced_audio_base64,
        "sample_rate": sr,
        "duration_seconds": len(audio) / sr
    }
```

## Music Analysis

### Beat Detection and Tempo Analysis

```python
import librosa
import numpy as np
from typing import List, Tuple

class MusicAnalyzer:
    def __init__(self):
        self.sample_rate = 22050

    def detect_beats(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect beats and estimate tempo"""
        # Extract tempo and beats
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=512
        )

        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)

        return beat_times, tempo

    def analyze_key_signature(self, audio: np.ndarray) -> str:
        """Analyze musical key signature"""
        # Extract chromagram
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)

        # Average chroma across time
        chroma_mean = np.mean(chroma, axis=1)

        # Key templates (major and minor)
        major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        # Find best matching key
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_correlation = -1
        best_key = 'C major'

        for i in range(12):
            # Test major
            major_corr = np.corrcoef(chroma_mean, np.roll(major_template, i))[0, 1]
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = f"{keys[i]} major"

            # Test minor
            minor_corr = np.corrcoef(chroma_mean, np.roll(minor_template, i))[0, 1]
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = f"{keys[i]} minor"

        return best_key

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features for music analysis"""
        # Compute spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))

        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        return {
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_bandwidth": float(spectral_bandwidth),
            "zero_crossing_rate": float(zero_crossing_rate),
            "mfcc_features": mfcc_means.tolist()
        }

async def analyze_music(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Music analysis endpoint"""
    # Decode input audio
    audio_base64 = inputs["audio_base64"]
    audio_data = base64.b64decode(audio_base64)

    # Load audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file.write(audio_data)
        temp_file.flush()
        audio, sr = librosa.load(temp_file.name, sr=22050)

    analyzer = MusicAnalyzer()

    # Perform analysis
    beat_times, tempo = analyzer.detect_beats(audio)
    key_signature = analyzer.analyze_key_signature(audio)
    spectral_features = analyzer.extract_spectral_features(audio)

    return {
        "tempo": float(tempo),
        "beat_count": len(beat_times),
        "beat_times": beat_times.tolist(),
        "key_signature": key_signature,
        "spectral_features": spectral_features,
        "duration_seconds": len(audio) / sr
    }
```

## Deployment Examples

### Speech Recognition Service

```python
# Deploy speech recognition chute
speech_chute = Chute(
    username="myuser",
    name="speech-recognition",
    image=audio_image,
    entry_file="speech_recognition.py",
    entry_point="transcribe_speech",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8),
    timeout_seconds=300,
    concurrency=8
)

# Usage
transcription_result = speech_chute.run({
    "audio_base64": "...",  # Base64 encoded audio
    "language": "en",
    "word_timestamps": True
})
print(f"Transcription: {transcription_result['text']}")
```

### Audio Enhancement Service

```python
# Deploy audio enhancement chute
enhancement_chute = Chute(
    username="myuser",
    name="audio-enhancement",
    image=audio_image,
    entry_file="audio_enhancement.py",
    entry_point="enhance_audio",
    node_selector=NodeSelector(
        gpu_count=0,  # CPU-only for audio processing),
    timeout_seconds=120,
    concurrency=10
)

# Usage
enhanced_result = enhancement_chute.run({
    "audio_base64": "...",  # Base64 encoded audio
    "options": {
        "reduce_noise": True,
        "normalize": True,
        "target_level": -20.0,
        "eq_bands": [
            {"frequency": 100, "gain": -3.0, "q": 1.0},
            {"frequency": 1000, "gain": 2.0, "q": 1.5},
            {"frequency": 8000, "gain": 1.0, "q": 1.0}
        ]
    }
})
```

## Real-time Audio Pipeline

### WebSocket Audio Streaming

```python
import asyncio
import websockets
import json
import numpy as np

class RealTimeAudioProcessor:
    def __init__(self):
        self.transcriber = WhisperTranscriber("base")
        self.enhancer = AudioEnhancer()
        self.analyzer = MusicAnalyzer()

    async def process_audio_stream(self, websocket, path):
        """Handle real-time audio WebSocket connection"""
        try:
            async for message in websocket:
                data = json.loads(message)

                if data["type"] == "audio_chunk":
                    # Process audio chunk
                    audio_data = base64.b64decode(data["audio_base64"])

                    # Convert to numpy array
                    audio = np.frombuffer(audio_data, dtype=np.float32)

                    # Process based on request type
                    if data.get("process_type") == "transcribe":
                        result = await self.transcribe_chunk(audio)
                    elif data.get("process_type") == "enhance":
                        result = await self.enhance_chunk(audio)
                    elif data.get("process_type") == "analyze":
                        result = await self.analyze_chunk(audio)

                    # Send result back
                    await websocket.send(json.dumps({
                        "type": "result",
                        "data": result
                    }))

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

    async def transcribe_chunk(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio chunk"""
        # Simple transcription for real-time processing
        if len(audio) > 0:
            # Convert to bytes for transcriber
            audio_bytes = audio.tobytes()
            request = TranscriptionRequest(
                audio_base64=base64.b64encode(audio_bytes).decode(),
                temperature=0.0
            )
            result = self.transcriber.transcribe_audio(audio_bytes, request)
            return result.dict()
        return {"text": "", "confidence": 0.0}

# Start WebSocket server
async def start_audio_server():
    processor = RealTimeAudioProcessor()

    server = await websockets.serve(
        processor.process_audio_stream,
        "0.0.0.0",
        8765
    )

    print("Audio processing server started on ws://0.0.0.0:8765")
    await server.wait_closed()

# Run the server
if __name__ == "__main__":
    asyncio.run(start_audio_server())
```

## Next Steps

- **[Music Generation](music-generation)** - Generate music and audio content
- **[Text-to-Speech](text-to-speech)** - Advanced speech synthesis
- **[Real-time Streaming](streaming-responses)** - Build streaming audio applications
- **[Custom Training](custom-training)** - Train custom audio models

For production audio processing pipelines, see the [Audio Infrastructure Guide](../guides/audio-infrastructure).
