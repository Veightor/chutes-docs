# Templates

**Templates** in Chutes are pre-built, optimized configurations for common AI workloads. They provide production-ready setups with just a few lines of code, handling complex configurations like Docker images, model loading, API endpoints, and hardware requirements.

## What are Templates?

Templates are factory functions that create complete Chute configurations for specific use cases:

- 🚀 **One-line deployment** of complex AI systems
- 🔧 **Pre-optimized configurations** for performance and cost
- 📦 **Batteries-included** with all necessary dependencies
- 🎯 **Best practices** built-in by default
- 🔄 **Customizable** for specific needs

## Available Templates

### Language Model Templates

#### VLLM Template

High-performance language model serving with OpenAI-compatible API.

```python
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)
```

#### SGLang Template

Structured generation for complex prompting and reasoning.

```python
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)
```

### Embedding Templates

#### Text Embeddings Inference (TEI)

Optimized text embedding generation.

```python
from chutes.chute.template.tei import build_tei_chute

chute = build_tei_chute(
    username="myuser",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=8)
)
```

### Image Generation Templates

#### Diffusion Template

Stable Diffusion and other diffusion model serving.

```python
from chutes.chute.template.diffusion import build_diffusion_chute

chute = build_diffusion_chute(
    username="myuser",
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=12)
)
```

## Template Categories

### 🗣️ Language Models

**Use Cases**: Text generation, chat, completion, code generation

- **VLLM**: Production-scale LLM serving
- **SGLang**: Complex reasoning and structured generation
- **Transformers**: Custom model implementations

### 🔤 Text Processing

**Use Cases**: Embeddings, classification, named entity recognition

- **TEI**: Fast embedding generation
- **Sentence Transformers**: Semantic similarity
- **BERT**: Classification and encoding

### 🎨 Image Generation

**Use Cases**: Image synthesis, editing, style transfer

- **Diffusion**: Stable Diffusion variants
- **GAN**: Generative adversarial networks
- **ControlNet**: Controlled image generation

### 🎵 Audio Processing

**Use Cases**: Speech recognition, text-to-speech, music generation

- **Whisper**: Speech-to-text
- **TTS**: Text-to-speech synthesis
- **MusicGen**: Music generation

### 🎬 Video Processing

**Use Cases**: Video analysis, generation, editing

- **Video Analysis**: Object detection, tracking
- **Video Generation**: Text-to-video models
- **Video Enhancement**: Upscaling, stabilization

## Template Benefits

### 1. **Instant Deployment**

```python
# Without templates (complex setup)
image = (
    Image(username="myuser", name="vllm-app", tag="1.0")
    .from_base("nvidia/cuda:12.1-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install vllm==0.2.0")
    .run_command("pip install transformers torch")
    # ... 50+ more lines of configuration
)

chute = Chute(
    username="myuser",
    name="llm-service",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)

@chute.on_startup()
async def load_model(self):
    # ... complex model loading logic

@chute.cord(public_api_path="/v1/chat/completions")
async def chat_completions(self, request: ChatRequest):
    # ... OpenAI API compatibility logic

# With templates (one line)
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)
```

### 2. **Production-Ready Defaults**

```python
# Templates include:
# ✅ Optimized Docker images
# ✅ Proper error handling
# ✅ Logging and monitoring
# ✅ Health checks
# ✅ Resource optimization
# ✅ Security best practices
```

### 3. **Hardware Optimization**

```python
# Templates automatically optimize for:
# - GPU memory usage
# - CPU utilization
# - Network throughput
# - Storage requirements
```

## Template Customization

### Basic Customization

```python
from chutes.chute.template.vllm import build_vllm_chute

# Customize standard parameters
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=24
    ),
    concurrency=8,
    tagline="Custom LLM API",
    readme="# My Custom LLM\nPowered by VLLM"
)
```

### Advanced Customization

```python
# Custom engine arguments
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    engine_args={
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 32,
        "temperature": 0.7
    }
)

# Custom Docker image
custom_image = (
    Image(username="myuser", name="custom-vllm", tag="1.0")
    .from_base("nvidia/cuda:12.1-devel-ubuntu22.04")
    .with_python("3.11")
    .run_command("pip install vllm==0.2.0")
    .run_command("pip install my-custom-package")
)

chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    image=custom_image
)
```

### Template Extension

```python
# Extend a template with custom functionality
base_chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
)

# Add custom endpoints
@base_chute.cord(public_api_path="/custom/analyze")
async def analyze_text(self, text: str) -> dict:
    # Custom text analysis logic
    return {"analysis": "custom_result"}

# Add custom startup logic
@base_chute.on_startup()
async def custom_initialization(self):
    # Additional setup
    self.custom_processor = CustomProcessor()
```

## Template Parameters

### Common Parameters

All templates support these standard parameters:

```python
def build_template_chute(
    username: str,              # Required: Your Chutes username
    model_name: str,           # Required: HuggingFace model name
    revision: str = "main",    # Git revision/branch
    node_selector: NodeSelector = None,  # Hardware requirements
    image: str | Image = None, # Custom Docker image
    tagline: str = "",         # Short description
    readme: str = "",          # Markdown documentation
    concurrency: int = 1,      # Concurrent requests per instance
    **kwargs                   # Template-specific options
)
```

### Template-Specific Parameters

#### VLLM Template

```python
build_vllm_chute(
    # Standard parameters...
    engine_args: dict = None,   # VLLM engine configuration
    trust_remote_code: bool = False,  # Allow remote code execution
    max_model_len: int = None,  # Maximum sequence length
    gpu_memory_utilization: float = 0.85,  # GPU memory usage
    max_num_seqs: int = 128     # Maximum concurrent sequences
)
```

#### Diffusion Template

```python
build_diffusion_chute(
    # Standard parameters...
    pipeline_type: str = "text2img",  # Pipeline type
    scheduler: str = "euler",         # Diffusion scheduler
    safety_checker: bool = True,      # Content safety
    guidance_scale: float = 7.5,      # CFG scale
    num_inference_steps: int = 50     # Generation steps
)
```

#### TEI Template

```python
build_tei_chute(
    # Standard parameters...
    pooling: str = "mean",      # Pooling strategy
    normalize: bool = True,     # Normalize embeddings
    batch_size: int = 32,       # Inference batch size
    max_length: int = 512       # Maximum input length
)
```

## Template Comparison

### Language Model Templates

| Template     | Best For               | Performance | Memory    | API               |
| ------------ | ---------------------- | ----------- | --------- | ----------------- |
| VLLM         | Production LLM serving | Highest     | Optimized | OpenAI-compatible |
| SGLang       | Complex reasoning      | High        | Standard  | Custom structured |
| Transformers | Custom implementations | Medium      | High      | Flexible          |

### Image Templates

| Template            | Best For                 | Speed  | Quality | Customization |
| ------------------- | ------------------------ | ------ | ------- | ------------- |
| Diffusion           | General image generation | Fast   | High    | Extensive     |
| Stable Diffusion XL | High-resolution images   | Medium | Highest | Good          |
| ControlNet          | Controlled generation    | Medium | High    | Specialized   |

## Creating Custom Templates

### Simple Template Function

```python
def build_custom_nlp_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    task_type: str = "classification"
) -> Chute:
    """Custom NLP template for classification and NER"""

    # Create custom image
    image = (
        Image(username=username, name="custom-nlp", tag="1.0")
        .from_base("nvidia/cuda:12.1-runtime-ubuntu22.04")
        .with_python("3.11")
        .run_command("pip install transformers torch scikit-learn")
    )

    # Create chute
    chute = Chute(
        username=username,
        name=f"nlp-{task_type}",
        image=image,
        node_selector=node_selector,
        tagline=f"Custom {task_type} service"
    )

    # Add model loading
    @chute.on_startup()
    async def load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Add API endpoint
    @chute.cord(public_api_path=f"/{task_type}")
    async def classify(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        return {"predictions": predictions.tolist()}

    return chute

# Use the custom template
custom_chute = build_custom_nlp_chute(
    username="myuser",
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=8),
    task_type="sentiment"
)
```

### Advanced Template with Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CustomNLPConfig:
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = True
    cache_size: int = 1000

def build_advanced_nlp_chute(
    username: str,
    model_name: str,
    node_selector: NodeSelector,
    config: CustomNLPConfig = None
) -> Chute:
    """Advanced NLP template with configuration"""

    if config is None:
        config = CustomNLPConfig()

    # Build image with config-specific optimizations
    image = (
        Image(username=username, name="advanced-nlp", tag="1.0")
        .from_base("nvidia/cuda:12.1-runtime-ubuntu22.04")
        .with_python("3.11")
        .run_command("pip install transformers torch accelerate")
    )

    if config.use_gpu:
        image = image.with_env("CUDA_VISIBLE_DEVICES", "0")

    chute = Chute(
        username=username,
        name="advanced-nlp",
        image=image,
        node_selector=node_selector
    )

    @chute.on_startup()
    async def setup(self):
        # Initialize with configuration
        self.config = config
        self.cache = {}  # Simple caching

        # Load model
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if config.use_gpu:
            self.model = self.model.cuda()

    @chute.cord(public_api_path="/process")
    async def process_text(self, texts: list[str]) -> dict:
        # Batch processing with configuration
        results = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            # Check cache
            cached_results = []
            new_texts = []

            for text in batch:
                if text in self.cache and len(self.cache) < self.config.cache_size:
                    cached_results.append(self.cache[text])
                else:
                    new_texts.append(text)

            # Process new texts
            if new_texts:
                inputs = self.tokenizer(
                    new_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                )

                if self.config.use_gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Cache results
                for text, output in zip(new_texts, outputs.last_hidden_state):
                    result = output.mean(dim=0).cpu().tolist()
                    self.cache[text] = result
                    cached_results.append(result)

            results.extend(cached_results)

        return {"embeddings": results, "count": len(results)}

    return chute
```

## Template Best Practices

### 1. **Use Appropriate Templates**

```python
# For LLM inference
vllm_chute = build_vllm_chute(...)

# For embedding generation
tei_chute = build_tei_chute(...)

# For image generation
diffusion_chute = build_diffusion_chute(...)
```

### 2. **Customize Hardware Requirements**

```python
# Small models
small_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8
)

# Large models
large_selector = NodeSelector(
    gpu_count=2,
    min_vram_gb_per_gpu=40
)
```

### 3. **Version Control Your Models**

```python
# Always specify revision
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main"  # or specific commit hash
)
```

### 4. **Document Your Deployments**

```python
chute = build_vllm_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    tagline="Customer service chatbot",
    readme="""
    # Customer Service Bot

    This chute provides automated customer service responses
    using DialoGPT-medium.

    ## Usage
    Send POST requests to `/v1/chat/completions`
    """
)
```

## Next Steps

- **[VLLM Template](/docs/templates/vllm)** - Detailed VLLM documentation
- **[Diffusion Template](/docs/templates/diffusion)** - Image generation guide
- **[TEI Template](/docs/templates/tei)** - Text embeddings guide
- **[Custom Templates Guide](/docs/guides/custom-templates)** - Build your own templates
