# Custom Templates Guide

This guide shows how to create reusable templates for common AI workflows, making it easy to deploy similar applications with different configurations.

## Overview

Custom templates in Chutes allow you to:

- **Standardize Deployments**: Create consistent deployment patterns
- **Reduce Code Duplication**: Reuse common configurations
- **Simplify Complex Setups**: Abstract away complexity for end users
- **Enable Team Collaboration**: Share best practices across teams

## Template Structure

### Basic Template Function

A template is a Python function that returns a configured Chute:

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector
from typing import Optional, Dict, Any

def build_text_classification_template(
    username: str,
    model_name: str,
    num_labels: int,
    node_selector: Optional[NodeSelector] = None,
    **kwargs
) -> Chute:
    """
    Template for text classification models

    Args:
        username: Chutes username
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        node_selector: Hardware requirements
        **kwargs: Additional chute configuration

    Returns:
        Configured Chute instance
    """

    # Default node selector
    if node_selector is None:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=8)

    # Build custom image
    image = (
        Image(
            username=username,
            name="text-classification",
            tag="latest",
            python_version="3.11"
        )
        .pip_install([
            "torch==2.1.0",
            "transformers==4.35.0",
            "datasets==2.14.0",
            "scikit-learn==1.3.0"
        ])
        .copy_files("./templates/text_classification", "/app")
    )

    # Create chute
    chute = Chute(
        username=username,
        name=f"text-classifier-{model_name.split('/')[-1]}",
        image=image,
        entry_file="classifier.py",
        entry_point="run",
        node_selector=node_selector,
        environment={
            "MODEL_NAME": model_name,
            "NUM_LABELS": str(num_labels)
        },
        timeout_seconds=300,
        concurrency=8,
        **kwargs
    )

    return chute

# Usage
classifier_chute = build_text_classification_template(
    username="myuser",
    model_name="bert-base-uncased",
    num_labels=3
)
```

## Advanced Template Examples

### Computer Vision Template

```python
def build_image_classification_template(
    username: str,
    model_name: str,
    image_size: int = 224,
    batch_size: int = 16,
    use_gpu: bool = True,
    **kwargs
) -> Chute:
    """Template for image classification models"""

    # Configure hardware based on requirements
    if use_gpu:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=12)
    else:
        node_selector = NodeSelector(
            gpu_count=0)

    # Build image with computer vision dependencies
    image = (
        Image(
            username=username,
            name="image-classification",
            tag=f"v{model_name.replace('/', '-')}",
            python_version="3.11"
        )
        .pip_install([
            "torch==2.1.0",
            "torchvision==0.16.0",
            "timm==0.9.7",
            "pillow==10.0.1",
            "opencv-python==4.8.1.78"
        ])
        .copy_files("./templates/image_classification", "/app")
    )

    chute = Chute(
        username=username,
        name=f"image-classifier-{model_name.split('/')[-1]}",
        image=image,
        entry_file="image_classifier.py",
        entry_point="run",
        node_selector=node_selector,
        environment={
            "MODEL_NAME": model_name,
            "IMAGE_SIZE": str(image_size),
            "BATCH_SIZE": str(batch_size)
        },
        timeout_seconds=600,
        concurrency=4,
        **kwargs
    )

    return chute

# Example implementation file: templates/image_classification/image_classifier.py
"""
import os
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Any
import base64
import io

class ImageClassifier:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME", "resnet50")
        self.image_size = int(os.environ.get("IMAGE_SIZE", "224"))
        self.batch_size = int(os.environ.get("BATCH_SIZE", "16"))

        # Load model
        self.model = timm.create_model(self.model_name, pretrained=True)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_b64: str) -> torch.Tensor:
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def predict(self, images: List[str]) -> List[Dict[str, Any]]:
        results = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]

            # Preprocess batch
            tensors = [self.preprocess_image(img) for img in batch]
            batch_tensor = torch.cat(tensors, dim=0)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Process results
            for j, probs in enumerate(probabilities):
                top5_probs, top5_indices = torch.topk(probs, 5)

                results.append({
                    "predictions": [
                        {
                            "class_id": int(idx),
                            "probability": float(prob)
                        }
                        for idx, prob in zip(top5_indices, top5_probs)
                    ]
                })

        return results

# Global classifier instance
classifier = ImageClassifier()

async def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
    images = inputs.get("images", [])
    if not images:
        return {"error": "No images provided"}

    results = classifier.predict(images)
    return {"results": results}
"""
```

### LLM Chat Template

```python
def build_llm_chat_template(
    username: str,
    model_name: str,
    max_length: int = 2048,
    temperature: float = 0.7,
    use_quantization: bool = False,
    **kwargs
) -> Chute:
    """Template for LLM chat applications"""

    # Determine hardware requirements based on model
    if "7b" in model_name.lower():
        vram_gb = 16 if not use_quantization else 8
    elif "13b" in model_name.lower():
        vram_gb = 24 if not use_quantization else 12
    elif "70b" in model_name.lower():
        vram_gb = 80 if not use_quantization else 40
    else:
        vram_gb = 16  # Default

    node_selector = NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=vram_gb)

    # Build image with LLM dependencies
    pip_packages = [
        "torch==2.1.0",
        "transformers==4.35.0",
        "accelerate==0.24.0"
    ]

    if use_quantization:
        pip_packages.append("bitsandbytes==0.41.0")

    image = (
        Image(
            username=username,
            name="llm-chat",
            tag=f"v{model_name.replace('/', '-')}",
            python_version="3.11"
        )
        .pip_install(pip_packages)
        .copy_files("./templates/llm_chat", "/app")
    )

    environment = {
        "MODEL_NAME": model_name,
        "MAX_LENGTH": str(max_length),
        "TEMPERATURE": str(temperature),
        "USE_QUANTIZATION": str(use_quantization).lower()
    }

    chute = Chute(
        username=username,
        name=f"llm-chat-{model_name.split('/')[-1]}",
        image=image,
        entry_file="chat_model.py",
        entry_point="run",
        node_selector=node_selector,
        environment=environment,
        timeout_seconds=300,
        concurrency=4,
        **kwargs
    )

    return chute
```

### Multi-Model Analysis Template

```python
def build_multi_model_analysis_template(
    username: str,
    models_config: Dict[str, Dict[str, Any]],
    enable_caching: bool = True,
    **kwargs
) -> Chute:
    """
    Template for multi-model analysis pipelines

    Args:
        username: Chutes username
        models_config: Dictionary of model configurations
            Example: {
                "sentiment": {"model": "cardiffnlp/twitter-roberta-base-sentiment"},
                "ner": {"model": "dbmdz/bert-large-cased-finetuned-conll03-english"},
                "classification": {"model": "facebook/bart-large-mnli"}
            }
        enable_caching: Whether to enable Redis caching
    """

    # Calculate resource requirements based on models
    total_models = len(models_config)
    estimated_vram = total_models * 4  # 4GB per model estimate

    node_selector = NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=max(16, estimated_vram)
    )

    # Build comprehensive image
    pip_packages = [
        "torch==2.1.0",
        "transformers==4.35.0",
        "datasets==2.14.0",
        "scikit-learn==1.3.0",
        "numpy==1.24.3",
        "asyncio-pool==0.6.0"
    ]

    if enable_caching:
        pip_packages.extend(["redis==5.0.0", "pickle5==0.0.12"])

    image = (
        Image(
            username=username,
            name="multi-model-analysis",
            tag="latest",
            python_version="3.11"
        )
        .pip_install(pip_packages)
        .copy_files("./templates/multi_model", "/app")
    )

    # Environment configuration
    environment = {
        "MODELS_CONFIG": json.dumps(models_config),
        "ENABLE_CACHING": str(enable_caching).lower()
    }

    if enable_caching:
        environment["REDIS_URL"] = "redis://localhost:6379"

    chute = Chute(
        username=username,
        name="multi-model-analyzer",
        image=image,
        entry_file="multi_analyzer.py",
        entry_point="run",
        node_selector=node_selector,
        environment=environment,
        timeout_seconds=600,
        concurrency=6,
        **kwargs
    )

    return chute

# Usage example
multi_model_chute = build_multi_model_analysis_template(
    username="myuser",
    models_config={
        "sentiment": {
            "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "task": "sentiment-analysis"
        },
        "ner": {
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "task": "ner"
        },
        "classification": {
            "model": "facebook/bart-large-mnli",
            "task": "zero-shot-classification"
        }
    },
    enable_caching=True
)
```

## Template Best Practices

### 1. Parameterization

Make templates flexible with good defaults:

```python
def build_flexible_template(
    username: str,
    model_name: str,
    # Required parameters
    task_type: str,

    # Optional parameters with sensible defaults
    python_version: str = "3.11",
    timeout_seconds: int = 300,
    concurrency: int = 8,
    enable_monitoring: bool = True,
    enable_caching: bool = True,
    auto_scale: bool = False,

    # Hardware configuration
    gpu_count: int = 1,
    min_vram_gb: int = 8,

    # Advanced configuration
    environment_vars: Optional[Dict[str, str]] = None,
    custom_pip_packages: Optional[List[str]] = None,

    **kwargs
) -> Chute:
    """Highly flexible template with many configuration options"""

    # Merge environment variables
    base_env = {
        "MODEL_NAME": model_name,
        "TASK_TYPE": task_type,
        "ENABLE_MONITORING": str(enable_monitoring).lower(),
        "ENABLE_CACHING": str(enable_caching).lower()
    }

    if environment_vars:
        base_env.update(environment_vars)

    # Build pip packages list
    base_packages = [
        "torch==2.1.0",
        "transformers==4.35.0"
    ]

    if enable_monitoring:
        base_packages.append("prometheus-client==0.18.0")

    if enable_caching:
        base_packages.append("redis==5.0.0")

    if custom_pip_packages:
        base_packages.extend(custom_pip_packages)

    # Configure node selector
    node_selector = NodeSelector(
        gpu_count=gpu_count,
        min_vram_gb_per_gpu=min_vram_gb)

    # Build image
    image = (
        Image(
            username=username,
            name=f"{task_type}-model",
            tag=model_name.replace("/", "-"),
            python_version=python_version
        )
        .pip_install(base_packages)
        .copy_files(f"./templates/{task_type}", "/app")
    )

    # Create chute
    chute = Chute(
        username=username,
        name=f"{task_type}-{model_name.split('/')[-1]}",
        image=image,
        entry_file="app.py",
        entry_point="run",
        node_selector=node_selector,
        environment=base_env,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        auto_scale=auto_scale,
        **kwargs
    )

    return chute
```

### 2. Template Validation

Add validation to prevent common errors:

```python
def validate_template_inputs(
    model_name: str,
    task_type: str,
    gpu_count: int,
    min_vram_gb: int
) -> None:
    """Validate template inputs"""

    # Validate model name format
    if "/" not in model_name:
        raise ValueError("model_name should be in format 'organization/model'")

    # Validate task type
    valid_tasks = ["classification", "ner", "generation", "embedding"]
    if task_type not in valid_tasks:
        raise ValueError(f"task_type must be one of {valid_tasks}")

    # Validate hardware requirements
    if gpu_count < 0 or gpu_count > 8:
        raise ValueError("gpu_count must be between 0 and 8")

    if min_vram_gb < 4 or min_vram_gb > 80:
        raise ValueError("min_vram_gb must be between 4 and 80")

    # Model-specific validation
    if "70b" in model_name.lower() and min_vram_gb < 40:
        raise ValueError("70B models require at least 40GB VRAM")

def build_validated_template(username: str, model_name: str, **kwargs) -> Chute:
    """Template with input validation"""

    # Extract and validate key parameters
    task_type = kwargs.get("task_type", "classification")
    gpu_count = kwargs.get("gpu_count", 1)
    min_vram_gb = kwargs.get("min_vram_gb", 8)

    validate_template_inputs(model_name, task_type, gpu_count, min_vram_gb)

    # Continue with template creation...
    return build_flexible_template(username, model_name, task_type, **kwargs)
```

### 3. Template Documentation

Document templates thoroughly:

```python
def build_documented_template(
    username: str,
    model_name: str,
    **kwargs
) -> Chute:
    """
    Production-ready template for ML model deployment

    This template provides a robust foundation for deploying machine learning
    models with monitoring, caching, and auto-scaling capabilities.

    Args:
        username (str): Your Chutes username
        model_name (str): HuggingFace model identifier (e.g., 'bert-base-uncased')

    Keyword Args:
        task_type (str): Type of ML task ('classification', 'ner', 'generation')
            Default: 'classification'
        gpu_count (int): Number of GPUs required (0-8)
            Default: 1
        min_vram_gb (int): Minimum VRAM per GPU in GB (4-80)
            Default: 8
        enable_monitoring (bool): Enable Prometheus metrics
            Default: True
        enable_caching (bool): Enable Redis caching
            Default: True
        auto_scale (bool): Enable auto-scaling
            Default: False

    Returns:
        Chute: Configured chute instance ready for deployment

    Example:
        >>> chute = build_documented_template(
        ...     username="myuser",
        ...     model_name="bert-base-uncased",
        ...     task_type="classification",
        ...     enable_monitoring=True,
        ...     auto_scale=True
        ... )
        >>> result = chute.deploy()

    Raises:
        ValueError: If invalid parameters are provided

    Note:
        This template automatically configures hardware requirements based on
        the model size. For 70B+ models, consider using multiple GPUs.
    """

    # Template implementation...
    pass
```

## Creating Template Packages

### Organizing Templates

Structure templates as reusable packages:

```
my_chutes_templates/
├── __init__.py
├── text/
│   ├── __init__.py
│   ├── classification.py
│   ├── generation.py
│   └── embedding.py
├── vision/
│   ├── __init__.py
│   ├── classification.py
│   ├── detection.py
│   └── segmentation.py
├── audio/
│   ├── __init__.py
│   ├── transcription.py
│   └── generation.py
└── templates/
    ├── text_classification/
    │   ├── app.py
    │   └── requirements.txt
    ├── image_classification/
    │   ├── app.py
    │   └── requirements.txt
    └── audio_transcription/
        ├── app.py
        └── requirements.txt
```

### Package Implementation

```python
# my_chutes_templates/__init__.py
from .text.classification import build_text_classification_template
from .text.generation import build_text_generation_template
from .vision.classification import build_image_classification_template

__all__ = [
    "build_text_classification_template",
    "build_text_generation_template",
    "build_image_classification_template"
]

__version__ = "1.0.0"

# my_chutes_templates/text/classification.py
from ..base import BaseTemplate

class TextClassificationTemplate(BaseTemplate):
    """Template for text classification models"""

    def __init__(self):
        super().__init__(
            template_name="text_classification",
            required_params=["model_name", "num_labels"],
            default_packages=[
                "torch==2.1.0",
                "transformers==4.35.0",
                "scikit-learn==1.3.0"
            ]
        )

    def build(self, username: str, **kwargs) -> Chute:
        return self._build_template(username, **kwargs)

def build_text_classification_template(username: str, **kwargs) -> Chute:
    """Convenience function for building text classification template"""
    template = TextClassificationTemplate()
    return template.build(username, **kwargs)
```

## Template Testing

### Unit Tests for Templates

```python
import unittest
from unittest.mock import patch, MagicMock
from my_chutes_templates import build_text_classification_template

class TestTextClassificationTemplate(unittest.TestCase):

    def test_template_creation(self):
        """Test basic template creation"""
        chute = build_text_classification_template(
            username="testuser",
            model_name="bert-base-uncased",
            num_labels=3
        )

        self.assertEqual(chute.username, "testuser")
        self.assertIn("bert-base-uncased", chute.name)
        self.assertEqual(chute.environment["NUM_LABELS"], "3")

    def test_invalid_parameters(self):
        """Test validation of invalid parameters"""
        with self.assertRaises(ValueError):
            build_text_classification_template(
                username="testuser",
                model_name="invalid-model",  # Invalid format
                num_labels=3
            )

    @patch('chutes.chute.Chute.deploy')
    def test_template_deployment(self, mock_deploy):
        """Test template deployment"""
        mock_deploy.return_value = {"status": "success"}

        chute = build_text_classification_template(
            username="testuser",
            model_name="bert-base-uncased",
            num_labels=3
        )

        result = chute.deploy()
        self.assertEqual(result["status"], "success")
        mock_deploy.assert_called_once()

if __name__ == "__main__":
    unittest.main()
```

## Next Steps

- **[Best Practices](best-practices)** - General deployment best practices
- **[Templates Guide](templates)** - Using existing templates
- **[Performance Optimization](performance)** - Optimize your custom templates

For advanced template development, see the [Template Development Guide](../advanced/template-development).
