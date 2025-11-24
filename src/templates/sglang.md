# SGLang Template

The **SGLang template** provides structured generation capabilities for complex prompting, reasoning, and multi-step AI workflows. SGLang (Structured Generation Language) excels at complex reasoning tasks and controlled text generation.

## What is SGLang?

SGLang is a domain-specific language for complex prompting and generation that provides:

- 🧠 **Structured reasoning** with multi-step prompts
- 🔄 **Control flow** for dynamic generation
- 📊 **State management** across generation steps
- 🎯 **Guided generation** with constraints
- 🔗 **Chain-of-thought** prompting patterns

## Quick Start

```python
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    )
)
```

This creates a complete SGLang deployment with:

- ✅ Structured generation engine
- ✅ Multi-step reasoning capabilities
- ✅ Custom prompting patterns
- ✅ State-aware generation
- ✅ Auto-scaling based on demand

## Function Reference

### `build_sglang_chute()`

```python
def build_sglang_chute(
    username: str,
    model_name: str,
    revision: str = "main",
    node_selector: NodeSelector = None,
    image: str | Image = None,
    tagline: str = "",
    readme: str = "",
    concurrency: int = 1,

    # SGLang-specific parameters
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    guidance_scale: float = 1.0,
    enable_sampling: bool = True,
    structured_output: bool = True,
    **kwargs
) -> Chute:
```

#### Required Parameters

- **`username`**: Your Chutes username
- **`model_name`**: HuggingFace model identifier

#### SGLang Configuration

- **`max_new_tokens`**: Maximum tokens to generate (default: 512)
- **`temperature`**: Sampling temperature (default: 0.7)
- **`top_p`**: Nucleus sampling parameter (default: 0.9)
- **`guidance_scale`**: Guidance strength for controlled generation (default: 1.0)
- **`enable_sampling`**: Enable probabilistic sampling (default: True)
- **`structured_output`**: Enable structured output formatting (default: True)

## Complete Example

```python
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

# Build SGLang chute for complex reasoning
chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    revision="main",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16
    ),
    tagline="Advanced reasoning with SGLang",
    readme="""
# Advanced Reasoning Engine

This chute provides structured generation capabilities using SGLang
for complex reasoning and multi-step AI workflows.

## Features
- Multi-step reasoning
- Structured output generation
- Chain-of-thought prompting
- Guided generation

## API Endpoints
- `/generate` - Basic text generation
- `/reason` - Multi-step reasoning
- `/structured` - Structured output generation
    """,

    # SGLang configuration
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.95,
    guidance_scale=1.2,
    structured_output=True
)
```

## API Endpoints

### Basic Generation

```bash
curl -X POST https://myuser-sglang-chute.chutes.ai/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Structured Reasoning

```bash
curl -X POST https://myuser-sglang-chute.chutes.ai/reason \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "What are the environmental impacts of renewable energy?",
    "steps": [
      "analyze_benefits",
      "identify_drawbacks",
      "compare_alternatives",
      "provide_conclusion"
    ]
  }'
```

### Chain-of-Thought

```bash
curl -X POST https://myuser-sglang-chute.chutes.ai/chain-of-thought \
  -H "Content-Type: application/json" \
  -d '{
    "question": "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "show_reasoning": true
  }'
```

## SGLang Programs

### Multi-Step Reasoning

```python
@sglang.function
def analyze_problem(s, problem):
    s += f"Problem: {problem}\n\n"
    s += "Let me think about this step by step:\n\n"

    s += "Step 1: Understanding the problem\n"
    s += sglang.gen("understanding", max_tokens=100)
    s += "\n\n"

    s += "Step 2: Identifying key factors\n"
    s += sglang.gen("factors", max_tokens=100)
    s += "\n\n"

    s += "Step 3: Analysis\n"
    s += sglang.gen("analysis", max_tokens=150)
    s += "\n\n"

    s += "Conclusion:\n"
    s += sglang.gen("conclusion", max_tokens=100)

    return s
```

### Structured Output

```python
@sglang.function
def extract_information(s, text):
    s += f"Text: {text}\n\n"
    s += "Extract the following information:\n\n"

    s += "Name: "
    s += sglang.gen("name", max_tokens=20, stop=["\n"])
    s += "\n"

    s += "Age: "
    s += sglang.gen("age", max_tokens=10, regex=r"\d+")
    s += "\n"

    s += "Occupation: "
    s += sglang.gen("occupation", max_tokens=30, stop=["\n"])
    s += "\n"

    s += "Summary: "
    s += sglang.gen("summary", max_tokens=100)

    return s
```

### Guided Generation

```python
@sglang.function
def generate_story(s, theme, character):
    s += f"Write a story about {character} with the theme of {theme}.\n\n"

    # Structured story generation
    s += "Title: "
    s += sglang.gen("title", max_tokens=20, stop=["\n"])
    s += "\n\n"

    s += "Setting: "
    s += sglang.gen("setting", max_tokens=50, stop=["\n"])
    s += "\n\n"

    s += "Plot:\n"

    for i in range(3):
        s += f"Chapter {i+1}: "
        s += sglang.gen(f"chapter_{i+1}", max_tokens=200)
        s += "\n\n"

    s += "Conclusion: "
    s += sglang.gen("conclusion", max_tokens=100)

    return s
```

## Advanced Features

### Custom Templates

```python
# Define custom reasoning template
reasoning_template = """
Problem: {problem}

Analysis Framework:
1. Context: What background information is relevant?
2. Constraints: What limitations or requirements exist?
3. Options: What are the possible approaches or solutions?
4. Evaluation: What are the pros and cons of each option?
5. Conclusion: What is the best approach and why?

Let me work through this systematically:
"""

chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    custom_templates={"reasoning": reasoning_template},
    guidance_scale=1.5  # Higher guidance for structured output
)
```

### Constraint-Based Generation

```python
# Configure constraints for specific output formats
chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    constraints={
        "json_format": True,
        "max_length": 500,
        "required_fields": ["summary", "key_points", "conclusion"],
        "stop_sequences": ["END", "STOP"]
    }
)
```

### Multi-Modal Reasoning

```python
# Enable multi-modal capabilities
chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    multimodal=True,
    vision_enabled=True,
    audio_enabled=False
)
```

## Model Recommendations

### Small Models (< 7B parameters)

```python
# Good for basic structured generation
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=8,
    include=["rtx4090", "rtx3090"]
)

# Recommended models:
# - microsoft/DialoGPT-medium
# - google/flan-t5-base
# - microsoft/DialoGPT-small
```

### Medium Models (7B - 13B parameters)

```python
# Optimal for complex reasoning
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=16,
    include=["rtx4090", "a100"]
)

# Recommended models:
# - microsoft/DialoGPT-large
# - google/flan-t5-large
# - meta-llama/Llama-2-7b-chat-hf
```

### Large Models (13B+ parameters)

```python
# Best for advanced reasoning
NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["a100", "h100"]
)

# Recommended models:
# - meta-llama/Llama-2-13b-chat-hf
# - microsoft/DialoGPT-xlarge
# - google/flan-ul2
```

## Use Cases

### 1. **Educational Tutoring**

```python
tutoring_chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    tagline="AI Tutor with structured explanations",
    custom_templates={
        "explanation": "Explain {topic} step by step with examples",
        "quiz": "Create 5 questions about {topic} with explanations"
    }
)
```

### 2. **Business Analysis**

```python
analysis_chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    structured_output=True,
    constraints={
        "format": "business_report",
        "sections": ["executive_summary", "analysis", "recommendations"]
    }
)
```

### 3. **Creative Writing**

```python
writing_chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    temperature=0.9,  # Higher creativity
    top_p=0.95,
    enable_sampling=True
)
```

### 4. **Code Generation**

```python
code_chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    temperature=0.3,  # Lower for more precise code
    structured_output=True,
    constraints={
        "language": "python",
        "include_comments": True,
        "include_tests": True
    }
)
```

## Performance Optimization

### Memory Optimization

```python
# Optimize for memory efficiency
chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    max_new_tokens=256,  # Limit generation length
    batch_size=4,        # Smaller batches
    gradient_checkpointing=True
)
```

### Speed Optimization

```python
# Optimize for speed
chute = build_sglang_chute(
    username="myuser",
    model_name="microsoft/DialoGPT-medium",
    temperature=0.0,     # Deterministic (faster)
    top_p=1.0,          # No nucleus sampling
    enable_caching=True, # Cache intermediate results
    compile_model=True   # JIT compilation
)
```

## Testing Your SGLang Chute

### Python Client

```python
import requests

# Test basic generation
response = requests.post(
    "https://myuser-sglang-chute.chutes.ai/generate",
    json={
        "prompt": "Analyze the benefits of renewable energy",
        "max_tokens": 300,
        "structured": True
    }
)

result = response.json()
print(result["generated_text"])
```

### Complex Reasoning Test

```python
# Test multi-step reasoning
response = requests.post(
    "https://myuser-sglang-chute.chutes.ai/reason",
    json={
        "problem": "Should companies adopt remote work policies?",
        "reasoning_steps": [
            "identify_stakeholders",
            "analyze_benefits",
            "analyze_drawbacks",
            "consider_implementation",
            "provide_recommendation"
        ]
    }
)

reasoning = response.json()
for step in reasoning["steps"]:
    print(f"{step['name']}: {step['output']}")
```

## Troubleshooting

### Common Issues

**Generation too slow?**

- Reduce `max_new_tokens`
- Lower `temperature` for deterministic output
- Disable sampling with `enable_sampling=False`

**Output not structured enough?**

- Increase `guidance_scale`
- Enable `structured_output=True`
- Add custom constraints

**Memory errors?**

- Reduce batch size
- Use smaller model
- Increase GPU VRAM requirements

**Inconsistent outputs?**

- Lower temperature for more deterministic results
- Use seed for reproducible generation
- Add stronger constraints

## Best Practices

### 1. **Template Design**

```python
# Good: Clear, structured templates
template = """
Task: {task}

Requirements:
- Be specific and detailed
- Provide examples
- Explain reasoning

Response:
"""

# Bad: Vague, unstructured
template = "Do {task}"
```

### 2. **Constraint Configuration**

```python
# Effective constraints
constraints = {
    "max_length": 500,
    "required_sections": ["introduction", "analysis", "conclusion"],
    "format": "markdown",
    "tone": "professional"
}
```

### 3. **Prompt Engineering**

```python
# Structure prompts for better results
def create_analysis_prompt(topic):
    return f"""
    Analyze the topic: {topic}

    Please structure your response as:
    1. Overview (2-3 sentences)
    2. Key factors (bullet points)
    3. Analysis (detailed explanation)
    4. Conclusion (summary and implications)

    Analysis:
    """
```

## Next Steps

- **[VLLM Template](/docs/templates/vllm)** - High-performance LLM serving
- **[Custom Templates Guide](/docs/guides/custom-templates)** - Build custom templates
- **[Advanced Prompting](/docs/guides/advanced-prompting)** - Prompt engineering techniques
- **[Multi-Model Workflows](/docs/guides/multi-model)** - Combine multiple models
