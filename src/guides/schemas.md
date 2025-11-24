# Input/Output Schemas with Pydantic

This guide covers how to use Pydantic for robust input/output validation in Chutes applications, enabling type safety, automatic API documentation, and data transformation.

## Overview

Pydantic schemas in Chutes provide:

- **Type Safety**: Automatic type validation and conversion
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Error Handling**: Clear validation error messages
- **Data Transformation**: Automatic serialization/deserialization
- **IDE Support**: Full autocomplete and type checking
- **Validation Rules**: Custom validators and constraints

## Basic Schema Definition

### Simple Input/Output Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze")
    language: Optional[str] = Field("auto", description="Language code (auto-detect if not specified)")
    options: Optional[List[str]] = Field(default=[], description="Additional processing options")

class AnalysisOutput(BaseModel):
    result: str = Field(..., description="Analysis result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    language_detected: Optional[str] = Field(None, description="Detected language code")
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

# Usage in chute
from chutes.chute import Chute

chute = Chute(username="myuser", name="text-analyzer")

@chute.cord(
    public_api_path="/analyze",
    method="POST",
    input_schema=TextInput,
    output_schema=AnalysisOutput
)
async def analyze_text(self, input_data: TextInput) -> AnalysisOutput:
    """Analyze text with full type safety."""

    # Input is automatically validated and typed
    text = input_data.text
    language = input_data.language
    options = input_data.options

    # Process text (example)
    result = f"Analyzed: {text[:50]}..."
    confidence = 0.95

    # Return validated output
    return AnalysisOutput(
        result=result,
        confidence=confidence,
        language_detected="en",
        processing_time=0.1
    )
```

### Advanced Field Validation

```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import Union, Literal
import re

class ImageGenerationInput(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Text prompt for image generation"
    )

    width: int = Field(
        512,
        ge=128,
        le=2048,
        multiple_of=64,  # Must be divisible by 64
        description="Image width in pixels"
    )

    height: int = Field(
        512,
        ge=128,
        le=2048,
        multiple_of=64,
        description="Image height in pixels"
    )

    steps: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of inference steps"
    )

    guidance_scale: float = Field(
        7.5,
        ge=1.0,
        le=20.0,
        description="Guidance scale for generation"
    )

    style: Literal["realistic", "artistic", "cartoon", "abstract"] = Field(
        "realistic",
        description="Image style"
    )

    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2**32-1,
        description="Random seed for reproducibility"
    )

    negative_prompt: Optional[str] = Field(
        None,
        max_length=500,
        description="Negative prompt to avoid certain elements"
    )

    @validator('prompt')
    def validate_prompt(cls, v):
        """Custom prompt validation."""
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())

        # Check for inappropriate content (example)
        forbidden_words = ['violence', 'harmful']
        if any(word in v.lower() for word in forbidden_words):
            raise ValueError('Prompt contains inappropriate content')

        return v

    @validator('width', 'height')
    def validate_dimensions(cls, v, field):
        """Validate image dimensions."""
        if v % 64 != 0:
            raise ValueError(f'{field.name} must be divisible by 64')
        return v

    @root_validator
    def validate_aspect_ratio(cls, values):
        """Validate overall aspect ratio."""
        width = values.get('width', 512)
        height = values.get('height', 512)

        aspect_ratio = width / height
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            raise ValueError('Extreme aspect ratios not supported (must be between 0.25 and 4)')

        return values

    class Config:
        # Generate example values for documentation
        schema_extra = {
            "example": {
                "prompt": "a beautiful sunset over mountains",
                "width": 1024,
                "height": 768,
                "steps": 25,
                "guidance_scale": 7.5,
                "style": "realistic",
                "seed": 42,
                "negative_prompt": "blurry, low quality"
            }
        }
```

## Complex Schema Patterns

### Nested Schemas

```python
from typing import List, Dict, Any
from enum import Enum

class ProcessingOptions(BaseModel):
    """Nested schema for processing options."""

    enable_caching: bool = Field(True, description="Enable result caching")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Processing timeout")
    parallel_processing: bool = Field(False, description="Enable parallel processing")

class ModelConfig(BaseModel):
    """Model configuration schema."""

    model_name: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum output tokens")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")

class BatchProcessingInput(BaseModel):
    """Complex input schema with nested structures."""

    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    model_config: ModelConfig = Field(..., description="Model configuration")
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions, description="Processing options")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('texts')
    def validate_texts(cls, v):
        """Validate text list."""
        # Check each text
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 5000:
                raise ValueError(f'Text at index {i} too long (max 5000 characters)')
        return v

class ProcessingResult(BaseModel):
    """Individual processing result."""

    input_text: str
    output_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = Field(..., gt=0)
    model_used: str

class BatchProcessingOutput(BaseModel):
    """Complex output schema."""

    results: List[ProcessingResult] = Field(..., description="Processing results")
    total_processed: int = Field(..., ge=0, description="Total items processed")
    total_time: float = Field(..., gt=0, description="Total processing time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")

    @validator('success_rate')
    def validate_success_rate(cls, v, values):
        """Validate success rate consistency."""
        results = values.get('results', [])
        total_processed = values.get('total_processed', 0)

        if total_processed > 0:
            expected_rate = len(results) / total_processed
            if abs(v - expected_rate) > 0.01:  # Allow small floating point errors
                raise ValueError('Success rate inconsistent with results')

        return v
```

### Union Types and Polymorphic Schemas

```python
from typing import Union
from pydantic import Field, discriminator

class TextTask(BaseModel):
    task_type: Literal["text"] = "text"
    text: str = Field(..., description="Input text")
    model: str = Field("gpt-3.5-turbo", description="Text model to use")

class ImageTask(BaseModel):
    task_type: Literal["image"] = "image"
    prompt: str = Field(..., description="Image generation prompt")
    width: int = Field(512, ge=128, le=2048)
    height: int = Field(512, ge=128, le=2048)

class AudioTask(BaseModel):
    task_type: Literal["audio"] = "audio"
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field("default", description="Voice to use")
    speed: float = Field(1.0, ge=0.5, le=2.0)

# Union type with discriminator
TaskInput = Union[TextTask, ImageTask, AudioTask]

class UniversalProcessingInput(BaseModel):
    """Schema supporting multiple task types."""

    task: TaskInput = Field(..., discriminator='task_type', description="Task to process")
    priority: int = Field(1, ge=1, le=5, description="Task priority")
    callback_url: Optional[str] = Field(None, description="Callback URL for results")

# Usage in endpoint
@chute.cord(
    public_api_path="/process",
    method="POST",
    input_schema=UniversalProcessingInput
)
async def process_universal(self, input_data: UniversalProcessingInput):
    """Process different types of tasks."""

    task = input_data.task

    if task.task_type == "text":
        # Type narrowing - IDE knows this is TextTask
        return await self.process_text(task.text, task.model)
    elif task.task_type == "image":
        return await self.process_image(task.prompt, task.width, task.height)
    elif task.task_type == "audio":
        return await self.process_audio(task.text, task.voice, task.speed)
```

## Advanced Validation Techniques

### Custom Validators

```python
from pydantic import validator, ValidationError
import base64
import mimetypes

class FileUploadSchema(BaseModel):
    """Schema for file upload validation."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    data: str = Field(..., description="Base64 encoded file data")
    max_size_mb: int = Field(10, ge=1, le=100, description="Maximum file size in MB")

    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Filename cannot be empty')

        # Check for path traversal
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename')

        return v.strip()

    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate MIME type."""
        allowed_types = [
            'image/jpeg', 'image/png', 'image/gif',
            'text/plain', 'application/pdf',
            'audio/mpeg', 'audio/wav'
        ]

        if v not in allowed_types:
            raise ValueError(f'Content type {v} not allowed')

        return v

    @validator('data')
    def validate_base64_data(cls, v, values):
        """Validate base64 data and size."""
        try:
            # Decode base64
            decoded = base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid base64 encoding')

        # Check file size
        max_size_mb = values.get('max_size_mb', 10)
        max_size_bytes = max_size_mb * 1024 * 1024

        if len(decoded) > max_size_bytes:
            raise ValueError(f'File size exceeds {max_size_mb}MB limit')

        # Validate content type matches data
        content_type = values.get('content_type')
        if content_type:
            # Simple validation - in practice, you'd use more sophisticated detection
            if content_type.startswith('image/') and not decoded[:10].startswith(b'\xff\xd8\xff'):
                if not (decoded[:8] == b'\x89PNG\r\n\x1a\n'):  # PNG header
                    raise ValueError('File content does not match declared type')

        return v

class ModelSelectionSchema(BaseModel):
    """Schema with model-specific validation."""

    model_name: str = Field(..., description="Model identifier")
    input_text: str = Field(..., description="Input text")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")

    @validator('parameters')
    def validate_model_parameters(cls, v, values):
        """Validate parameters based on model."""
        model_name = values.get('model_name', '')

        # Model-specific parameter validation
        if 'gpt' in model_name.lower():
            # GPT models
            if 'temperature' in v and not (0.0 <= v['temperature'] <= 2.0):
                raise ValueError('Temperature must be between 0.0 and 2.0 for GPT models')
            if 'max_tokens' in v and not (1 <= v['max_tokens'] <= 4096):
                raise ValueError('max_tokens must be between 1 and 4096 for GPT models')

        elif 'bert' in model_name.lower():
            # BERT models don't use temperature
            if 'temperature' in v:
                raise ValueError('Temperature parameter not applicable for BERT models')

        return v
```

### Dynamic Validation

```python
from typing import Callable, Any
import inspect

class DynamicValidationSchema(BaseModel):
    """Schema with dynamic validation rules."""

    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(..., description="Operation parameters")

    @validator('parameters')
    def validate_parameters_for_operation(cls, v, values):
        """Validate parameters based on operation type."""
        operation = values.get('operation')

        validation_rules = {
            'sentiment_analysis': {
                'required': ['text'],
                'optional': ['model', 'language'],
                'types': {'text': str, 'model': str, 'language': str}
            },
            'image_generation': {
                'required': ['prompt'],
                'optional': ['width', 'height', 'steps'],
                'types': {'prompt': str, 'width': int, 'height': int, 'steps': int},
                'ranges': {'width': (128, 2048), 'height': (128, 2048), 'steps': (1, 100)}
            },
            'translation': {
                'required': ['text', 'target_language'],
                'optional': ['source_language'],
                'types': {'text': str, 'target_language': str, 'source_language': str}
            }
        }

        if operation not in validation_rules:
            raise ValueError(f'Unknown operation: {operation}')

        rules = validation_rules[operation]

        # Check required parameters
        for param in rules['required']:
            if param not in v:
                raise ValueError(f'Missing required parameter: {param}')

        # Check parameter types
        for param, value in v.items():
            if param in rules['types']:
                expected_type = rules['types'][param]
                if not isinstance(value, expected_type):
                    raise ValueError(f'Parameter {param} must be of type {expected_type.__name__}')

        # Check ranges
        if 'ranges' in rules:
            for param, (min_val, max_val) in rules['ranges'].items():
                if param in v:
                    if not (min_val <= v[param] <= max_val):
                        raise ValueError(f'Parameter {param} must be between {min_val} and {max_val}')

        return v

class ConfigurableSchema(BaseModel):
    """Schema that can be configured at runtime."""

    class Config:
        extra = "forbid"  # Don't allow extra fields by default

    @classmethod
    def create_with_extra_fields(cls, extra_fields: Dict[str, Any]):
        """Create schema variant that allows specific extra fields."""

        class DynamicSchema(cls):
            class Config:
                extra = "allow"

            @validator('*', pre=True, allow_reuse=True)
            def validate_extra_fields(cls, v, field):
                if field.name in extra_fields:
                    # Validate against provided rules
                    field_rules = extra_fields[field.name]
                    if 'type' in field_rules and not isinstance(v, field_rules['type']):
                        raise ValueError(f'Field {field.name} must be of type {field_rules["type"].__name__}')
                    if 'range' in field_rules:
                        min_val, max_val = field_rules['range']
                        if not (min_val <= v <= max_val):
                            raise ValueError(f'Field {field.name} must be between {min_val} and {max_val}')
                return v

        return DynamicSchema
```

## Error Handling and User-Friendly Messages

### Custom Error Messages

```python
from pydantic import ValidationError, Field
from typing import List, Dict

class UserFriendlySchema(BaseModel):
    """Schema with user-friendly error messages."""

    email: str = Field(
        ...,
        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        description="Valid email address",
        error_msg="Please enter a valid email address (e.g., user@example.com)"
    )

    age: int = Field(
        ...,
        ge=13,
        le=120,
        description="Age in years",
        error_msg="Age must be between 13 and 120 years"
    )

    password: str = Field(
        ...,
        min_length=8,
        description="Password (minimum 8 characters)",
        error_msg="Password must be at least 8 characters long"
    )

    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength with clear messages."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

def format_validation_errors(e: ValidationError) -> Dict[str, List[str]]:
    """Format validation errors for user-friendly display."""

    error_dict = {}

    for error in e.errors():
        field_path = " -> ".join(str(loc) for loc in error['loc'])
        error_msg = error['msg']

        # Customize error messages
        if error['type'] == 'value_error.missing':
            error_msg = "This field is required"
        elif error['type'] == 'type_error.str':
            error_msg = "This field must be text"
        elif error['type'] == 'type_error.integer':
            error_msg = "This field must be a number"
        elif error['type'] == 'value_error.number.not_ge':
            error_msg = f"This field must be at least {error['ctx']['limit_value']}"
        elif error['type'] == 'value_error.number.not_le':
            error_msg = f"This field must be at most {error['ctx']['limit_value']}"

        if field_path not in error_dict:
            error_dict[field_path] = []
        error_dict[field_path].append(error_msg)

    return error_dict

# Usage in endpoint
@chute.cord(public_api_path="/register", method="POST")
async def register_user(self, input_data: UserFriendlySchema):
    """Register user with friendly error handling."""
    try:
        # Process registration
        return {"message": "Registration successful"}
    except ValidationError as e:
        formatted_errors = format_validation_errors(e)
        raise HTTPException(status_code=422, detail=formatted_errors)
```

### Validation Error Recovery

```python
from typing import Union, Optional

class FlexibleInputSchema(BaseModel):
    """Schema that attempts to recover from validation errors."""

    text: str = Field(..., description="Input text")
    confidence_threshold: Union[float, str] = Field(0.5, description="Confidence threshold")
    max_results: Union[int, str] = Field(10, description="Maximum number of results")

    @validator('confidence_threshold', pre=True)
    def parse_confidence_threshold(cls, v):
        """Attempt to parse confidence threshold from string."""
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                raise ValueError('Confidence threshold must be a number between 0 and 1')

        if not isinstance(v, (int, float)):
            raise ValueError('Confidence threshold must be a number')

        if not (0.0 <= v <= 1.0):
            raise ValueError('Confidence threshold must be between 0 and 1')

        return float(v)

    @validator('max_results', pre=True)
    def parse_max_results(cls, v):
        """Attempt to parse max_results from string."""
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                raise ValueError('Max results must be a positive integer')

        if not isinstance(v, int):
            raise ValueError('Max results must be an integer')

        if v <= 0:
            raise ValueError('Max results must be positive')

        if v > 100:
            v = 100  # Auto-correct to maximum allowed

        return v

class AutoCorrectingSchema(BaseModel):
    """Schema that auto-corrects common input errors."""

    text: str = Field(..., description="Input text")
    language: str = Field("auto", description="Language code")

    @validator('text', pre=True)
    def clean_text(cls, v):
        """Clean and normalize text input."""
        if not isinstance(v, str):
            v = str(v)

        # Normalize whitespace
        v = re.sub(r'\s+', ' ', v.strip())

        # Remove common problematic characters
        v = v.replace('\x00', '')  # Remove null bytes
        v = v.replace('\ufeff', '')  # Remove BOM

        if len(v) == 0:
            raise ValueError('Text cannot be empty after cleaning')

        return v

    @validator('language', pre=True)
    def normalize_language(cls, v):
        """Normalize language codes."""
        if not isinstance(v, str):
            v = str(v)

        v = v.lower().strip()

        # Common language code mappings
        language_mappings = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'chinese': 'zh',
            'japanese': 'ja',
            'korean': 'ko',
            'auto-detect': 'auto',
            'automatic': 'auto'
        }

        if v in language_mappings:
            v = language_mappings[v]

        # Validate language code format
        if v != 'auto' and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError(f'Invalid language code: {v}')

        return v
```

## Schema Documentation and Examples

### Comprehensive Documentation

```python
class DocumentedAPISchema(BaseModel):
    """Fully documented API schema with examples."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text prompt for AI processing",
        example="Generate a creative story about space exploration"
    )

    model: str = Field(
        "gpt-3.5-turbo",
        description="AI model to use for processing",
        example="gpt-4",
        regex=r'^(gpt-3\.5-turbo|gpt-4|claude-2)$'
    )

    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Controls randomness in the output. Higher values make output more random.",
        example=0.8
    )

    max_tokens: int = Field(
        100,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate",
        example=250
    )

    stop_sequences: Optional[List[str]] = Field(
        None,
        max_items=4,
        description="List of sequences where generation should stop",
        example=[".", "!", "?"]
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a haiku about artificial intelligence",
                "model": "gpt-3.5-turbo",
                "temperature": 0.8,
                "max_tokens": 50,
                "stop_sequences": ["\n\n"]
            },
            "examples": {
                "creative_writing": {
                    "summary": "Creative writing example",
                    "value": {
                        "prompt": "Write a short story about a robot discovering emotions",
                        "model": "gpt-4",
                        "temperature": 0.9,
                        "max_tokens": 500
                    }
                },
                "technical_explanation": {
                    "summary": "Technical explanation example",
                    "value": {
                        "prompt": "Explain how neural networks work",
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.3,
                        "max_tokens": 300
                    }
                }
            }
        }

class ResponseSchema(BaseModel):
    """Well-documented response schema."""

    generated_text: str = Field(
        ...,
        description="The generated text output from the AI model",
        example="Artificial intelligence learns,\nProcessing data endlessly,\nFuture unfolds bright."
    )

    model_used: str = Field(
        ...,
        description="The actual model used for generation",
        example="gpt-3.5-turbo"
    )

    tokens_used: int = Field(
        ...,
        ge=0,
        description="Number of tokens consumed in generation",
        example=32
    )

    processing_time: float = Field(
        ...,
        gt=0,
        description="Time taken to process the request in seconds",
        example=1.25
    )

    finish_reason: Literal["completed", "max_tokens", "stop_sequence"] = Field(
        ...,
        description="Reason why generation finished",
        example="completed"
    )
```

### Schema Testing and Validation

```python
import pytest
from pydantic import ValidationError

class SchemaTestSuite:
    """Test suite for schema validation."""

    @staticmethod
    def test_valid_inputs():
        """Test valid input scenarios."""

        # Test basic valid input
        valid_data = {
            "prompt": "Hello world",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 100
        }

        schema = DocumentedAPISchema(**valid_data)
        assert schema.prompt == "Hello world"
        assert schema.temperature == 0.7

        # Test with optional fields
        valid_with_optional = {
            "prompt": "Test prompt",
            "stop_sequences": [".", "!"]
        }

        schema2 = DocumentedAPISchema(**valid_with_optional)
        assert schema2.model == "gpt-3.5-turbo"  # Default value
        assert schema2.stop_sequences == [".", "!"]

    @staticmethod
    def test_invalid_inputs():
        """Test invalid input scenarios."""

        # Test missing required field
        with pytest.raises(ValidationError) as exc_info:
            DocumentedAPISchema(model="gpt-4")

        errors = exc_info.value.errors()
        assert any(error['type'] == 'value_error.missing' for error in errors)

        # Test invalid temperature
        with pytest.raises(ValidationError) as exc_info:
            DocumentedAPISchema(prompt="test", temperature=3.0)

        errors = exc_info.value.errors()
        assert any('temperature' in str(error['loc']) for error in errors)

        # Test invalid model
        with pytest.raises(ValidationError) as exc_info:
            DocumentedAPISchema(prompt="test", model="invalid-model")

        errors = exc_info.value.errors()
        assert any('regex' in error['type'] for error in errors)

    @staticmethod
    def test_edge_cases():
        """Test edge cases and boundary conditions."""

        # Test minimum values
        min_data = {
            "prompt": "a",  # Minimum length
            "temperature": 0.0,
            "max_tokens": 1
        }

        schema = DocumentedAPISchema(**min_data)
        assert schema.temperature == 0.0

        # Test maximum values
        max_data = {
            "prompt": "x" * 1000,  # Maximum length
            "temperature": 2.0,
            "max_tokens": 4096
        }

        schema = DocumentedAPISchema(**max_data)
        assert len(schema.prompt) == 1000

# Run tests
if __name__ == "__main__":
    test_suite = SchemaTestSuite()
    test_suite.test_valid_inputs()
    test_suite.test_invalid_inputs()
    test_suite.test_edge_cases()
    print("All schema tests passed!")
```

## Performance and Best Practices

### Schema Performance Optimization

```python
from pydantic import BaseModel, Field, validator
from typing import ClassVar

class OptimizedSchema(BaseModel):
    """Performance-optimized schema."""

    # Use ClassVar for constants to avoid creating fields
    MAX_TEXT_LENGTH: ClassVar[int] = 5000
    ALLOWED_MODELS: ClassVar[set] = {"gpt-3.5-turbo", "gpt-4", "claude-2"}

    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    model: str = Field("gpt-3.5-turbo")

    @validator('model')
    def validate_model(cls, v):
        """Fast model validation using set lookup."""
        if v not in cls.ALLOWED_MODELS:
            raise ValueError(f'Model must be one of: {", ".join(cls.ALLOWED_MODELS)}')
        return v

    class Config:
        # Performance optimizations
        validate_assignment = False  # Don't validate on assignment
        allow_reuse = True  # Allow validator reuse
        str_strip_whitespace = True  # Auto-strip strings
        anystr_lower = False  # Don't auto-lowercase

class CachedValidationSchema(BaseModel):
    """Schema with cached validation results."""

    _validation_cache: ClassVar[Dict[str, bool]] = {}

    data: str = Field(...)

    @validator('data')
    def validate_with_cache(cls, v):
        """Use caching for expensive validation."""

        # Check cache first
        if v in cls._validation_cache:
            if not cls._validation_cache[v]:
                raise ValueError('Cached validation failed')
            return v

        # Perform expensive validation
        is_valid = cls._expensive_validation(v)

        # Cache result
        cls._validation_cache[v] = is_valid

        if not is_valid:
            raise ValueError('Validation failed')

        return v

    @staticmethod
    def _expensive_validation(data: str) -> bool:
        """Simulate expensive validation."""
        # This would be your actual expensive validation logic
        return len(data) > 0 and not any(char in data for char in ['<', '>', '&'])
```

### Schema Composition and Reuse

```python
from abc import ABC
from typing import Generic, TypeVar

# Base schemas for reuse
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class PaginationMixin(BaseModel):
    """Mixin for pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")

class MetadataMixin(BaseModel):
    """Mixin for metadata fields."""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list, max_items=10)

# Composed schemas
class UserInput(MetadataMixin):
    """User input with metadata support."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')

class PaginatedResponse(Generic[T], TimestampMixin, PaginationMixin):
    """Generic paginated response."""
    items: List[T] = Field(..., description="Response items")
    total: int = Field(..., ge=0, description="Total number of items")
    has_next: bool = Field(..., description="Whether there are more pages")

# Usage
T = TypeVar('T')

class ProcessingResult(BaseModel):
    result: str
    confidence: float

# Create specific paginated response
PaginatedProcessingResponse = PaginatedResponse[ProcessingResult]
```

## Next Steps

- **API Documentation**: Generate comprehensive API docs from schemas
- **Client Generation**: Auto-generate typed clients from schemas
- **Database Integration**: Connect schemas with ORMs and databases
- **Testing Strategies**: Implement comprehensive schema testing

For more advanced topics, see:

- [Error Handling Guide](error-handling)
- [Custom Chutes Guide](custom-chutes)
