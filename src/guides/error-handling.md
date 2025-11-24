# Error Handling and Resilience

This guide covers comprehensive error handling strategies for Chutes applications, ensuring robust, production-ready AI services that gracefully handle failures and provide meaningful feedback.

## Overview

Effective error handling in Chutes includes:

- **Graceful Degradation**: Handle failures without complete system breakdown
- **User-Friendly Messages**: Provide clear, actionable error information
- **Logging and Monitoring**: Track errors for debugging and improvement
- **Retry Strategies**: Automatically recover from transient failures
- **Circuit Breakers**: Prevent cascading failures
- **Fallback Mechanisms**: Provide alternative responses when primary methods fail

## Error Types and Classification

### AI Model Errors

```python
from enum import Enum
from typing import Optional, Dict, Any
import logging
from datetime import datetime

class AIErrorType(Enum):
    """Classification of AI-specific errors."""
    MODEL_LOADING_FAILED = "model_loading_failed"
    INFERENCE_TIMEOUT = "inference_timeout"
    OUT_OF_MEMORY = "out_of_memory"
    INVALID_INPUT = "invalid_input"
    MODEL_OVERLOADED = "model_overloaded"
    GENERATION_FAILED = "generation_failed"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"

class ModelError(Exception):
    """Base exception for model-related errors."""

    def __init__(
        self,
        message: str,
        error_type: AIErrorType,
        details: Optional[Dict[str, Any]] = None,
        is_retryable: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.is_retryable = is_retryable
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error": self.message,
            "error_type": self.error_type.value,
            "details": self.details,
            "is_retryable": self.is_retryable,
            "timestamp": self.timestamp.isoformat()
        }

class OutOfMemoryError(ModelError):
    """GPU/CPU memory exhaustion error."""

    def __init__(self, memory_used: Optional[int] = None, memory_available: Optional[int] = None):
        details = {}
        if memory_used is not None:
            details["memory_used_mb"] = memory_used
        if memory_available is not None:
            details["memory_available_mb"] = memory_available

        super().__init__(
            "Model inference failed due to insufficient memory",
            AIErrorType.OUT_OF_MEMORY,
            details=details,
            is_retryable=False
        )

class ContextLengthError(ModelError):
    """Input context too long for model."""

    def __init__(self, input_length: int, max_length: int):
        super().__init__(
            f"Input length ({input_length}) exceeds model's maximum context length ({max_length})",
            AIErrorType.CONTEXT_LENGTH_EXCEEDED,
            details={
                "input_length": input_length,
                "max_length": max_length,
                "suggestion": "Reduce input length or use a model with larger context window"
            },
            is_retryable=False
        )

class InferenceTimeoutError(ModelError):
    """Model inference timeout."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            f"Model inference timed out after {timeout_seconds} seconds",
            AIErrorType.INFERENCE_TIMEOUT,
            details={"timeout_seconds": timeout_seconds},
            is_retryable=True
        )
```

### Input Validation Errors

```python
from pydantic import ValidationError
from fastapi import HTTPException

class ValidationErrorHandler:
    """Handle and format validation errors."""

    @staticmethod
    def format_pydantic_error(validation_error: ValidationError) -> Dict[str, Any]:
        """Format Pydantic validation error for user-friendly display."""

        formatted_errors = {}

        for error in validation_error.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_type = error['type']

            # Create user-friendly error messages
            if error_type == 'value_error.missing':
                message = "This field is required"
            elif error_type == 'type_error.str':
                message = "This field must be text"
            elif error_type == 'type_error.integer':
                message = "This field must be a whole number"
            elif error_type == 'type_error.float':
                message = "This field must be a number"
            elif error_type == 'value_error.number.not_ge':
                limit = error['ctx']['limit_value']
                message = f"This field must be at least {limit}"
            elif error_type == 'value_error.number.not_le':
                limit = error['ctx']['limit_value']
                message = f"This field must be at most {limit}"
            elif error_type == 'value_error.str.regex':
                message = "This field has an invalid format"
            elif error_type == 'value_error.list.min_items':
                min_items = error['ctx']['limit_value']
                message = f"This list must have at least {min_items} items"
            elif error_type == 'value_error.list.max_items':
                max_items = error['ctx']['limit_value']
                message = f"This list can have at most {max_items} items"
            else:
                message = error['msg']

            if field_path not in formatted_errors:
                formatted_errors[field_path] = []
            formatted_errors[field_path].append(message)

        return {
            "error": "Validation failed",
            "error_type": "validation_error",
            "field_errors": formatted_errors,
            "is_retryable": False
        }

    @staticmethod
    def create_http_exception(validation_error: ValidationError) -> HTTPException:
        """Create HTTP exception from validation error."""
        formatted_error = ValidationErrorHandler.format_pydantic_error(validation_error)

        return HTTPException(
            status_code=422,
            detail=formatted_error
        )

class InputSanitizer:
    """Sanitize and validate inputs with error handling."""

    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """Sanitize text input with error handling."""
        if not isinstance(text, str):
            raise ValueError("Input must be text")

        # Remove null bytes and control characters
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Check length
        if len(sanitized) > max_length:
            raise ValueError(f"Input text too long (max {max_length} characters)")

        if len(sanitized.strip()) == 0:
            raise ValueError("Input text cannot be empty")

        return sanitized.strip()

    @staticmethod
    def validate_file_upload(file_data: bytes, allowed_types: list, max_size_mb: int = 10):
        """Validate file upload with comprehensive error checking."""

        # Check size
        if len(file_data) > max_size_mb * 1024 * 1024:
            raise ValueError(f"File too large (max {max_size_mb}MB)")

        # Check if empty
        if len(file_data) == 0:
            raise ValueError("File is empty")

        # Basic file type detection
        file_signatures = {
            b'\xff\xd8\xff': 'image/jpeg',
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'GIF87a': 'image/gif',
            b'GIF89a': 'image/gif',
            b'%PDF': 'application/pdf'
        }

        detected_type = None
        for signature, mime_type in file_signatures.items():
            if file_data.startswith(signature):
                detected_type = mime_type
                break

        if detected_type not in allowed_types:
            raise ValueError(f"File type not allowed. Allowed types: {', '.join(allowed_types)}")

        return detected_type
```

## Error Handling Decorators

### Retry Mechanisms

```python
import asyncio
import functools
from typing import Callable, Type, Tuple, Union
import random

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception)
):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)

                    logging.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    await asyncio.sleep(delay)

                except Exception as e:
                    # Non-retryable exception
                    logging.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise

            # All retries exhausted
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Handle synchronous functions
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def circuit_breaker(
    failure_threshold: int = 5,
    timeout_duration: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """Circuit breaker decorator to prevent cascading failures."""

    def decorator(func: Callable):
        # Shared state across all calls
        state = {
            'failures': 0,
            'last_failure_time': None,
            'state': 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        }

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            now = datetime.now().timestamp()

            # Check if circuit should transition to HALF_OPEN
            if (state['state'] == 'OPEN' and
                state['last_failure_time'] and
                now - state['last_failure_time'] > timeout_duration):
                state['state'] = 'HALF_OPEN'
                logging.info(f"Circuit breaker for {func.__name__} is now HALF_OPEN")

            # Reject if circuit is OPEN
            if state['state'] == 'OPEN':
                raise ModelError(
                    f"Circuit breaker is OPEN for {func.__name__}",
                    AIErrorType.MODEL_OVERLOADED,
                    details={'circuit_state': 'OPEN'},
                    is_retryable=True
                )

            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

                # Success - reset circuit if it was HALF_OPEN
                if state['state'] == 'HALF_OPEN':
                    state['state'] = 'CLOSED'
                    state['failures'] = 0
                    logging.info(f"Circuit breaker for {func.__name__} is now CLOSED")

                return result

            except expected_exception as e:
                state['failures'] += 1
                state['last_failure_time'] = now

                # Open circuit if threshold exceeded
                if state['failures'] >= failure_threshold:
                    state['state'] = 'OPEN'
                    logging.error(f"Circuit breaker for {func.__name__} is now OPEN")

                raise

        return wrapper

    return decorator

# Usage examples
@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    retryable_exceptions=(InferenceTimeoutError, ModelError)
)
@circuit_breaker(
    failure_threshold=5,
    timeout_duration=30.0,
    expected_exception=ModelError
)
async def robust_model_inference(self, input_data: str) -> str:
    """Model inference with retry and circuit breaker protection."""
    try:
        result = await self.model.generate(input_data)
        return result
    except torch.cuda.OutOfMemoryError:
        raise OutOfMemoryError()
    except TimeoutError:
        raise InferenceTimeoutError(30.0)
```

### Error Context Management

```python
import contextlib
from typing import Optional, Dict, Any, List

class ErrorContext:
    """Manage error context and correlation across operations."""

    def __init__(self):
        self.context_stack: List[Dict[str, Any]] = []
        self.correlation_id: Optional[str] = None

    def push_context(self, **kwargs):
        """Add context information."""
        self.context_stack.append(kwargs)

    def pop_context(self):
        """Remove last context."""
        if self.context_stack:
            self.context_stack.pop()

    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context information."""
        context = {}
        for ctx in self.context_stack:
            context.update(ctx)

        if self.correlation_id:
            context['correlation_id'] = self.correlation_id

        return context

    @contextlib.contextmanager
    def operation_context(self, **kwargs):
        """Context manager for operation-specific error context."""
        self.push_context(**kwargs)
        try:
            yield self
        finally:
            self.pop_context()

class ContextualError(Exception):
    """Exception that includes context information."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.message = message
        self.context = context.get_full_context() if context else {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API response."""
        return {
            "error": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }

def with_error_context(error_context: ErrorContext):
    """Decorator to add error context to functions."""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                # Wrap exception with context
                if not isinstance(e, ContextualError):
                    raise ContextualError(str(e), error_context) from e
                raise

        return wrapper

    return decorator

# Usage
error_context = ErrorContext()
error_context.correlation_id = "req-12345"

@with_error_context(error_context)
async def process_with_context(self, data: str):
    """Process data with error context tracking."""

    with error_context.operation_context(operation="preprocessing", input_size=len(data)):
        # Preprocessing step
        cleaned_data = self.preprocess(data)

    with error_context.operation_context(operation="inference", model="gpt-3.5-turbo"):
        # Inference step
        result = await self.model.generate(cleaned_data)

    return result
```

## Centralized Error Handling

### Error Handler Class

```python
import traceback
import sys
from typing import Union, Optional

class CentralizedErrorHandler:
    """Centralized error handling for Chutes applications."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.error_history = []

    async def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle error and return appropriate response."""

        context = context or {}
        error_type = type(error).__name__

        # Track error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Create error record
        error_record = {
            "error_type": error_type,
            "message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc() if self.logger.level <= logging.DEBUG else None
        }

        # Store in history (limited size)
        self.error_history.append(error_record)
        if len(self.error_history) > 1000:
            self.error_history.pop(0)

        # Log error
        self.logger.error(
            f"Error in {context.get('operation', 'unknown')}: {str(error)}",
            extra={
                "error_type": error_type,
                "context": context,
                "correlation_id": context.get("correlation_id")
            }
        )

        # Create user-facing response
        if isinstance(error, ModelError):
            response = error.to_dict()
        elif isinstance(error, ValidationError):
            response = ValidationErrorHandler.format_pydantic_error(error)
        elif isinstance(error, ContextualError):
            response = error.to_dict()
        else:
            # Generic error handling
            response = {
                "error": user_message or "An unexpected error occurred",
                "error_type": "internal_error",
                "is_retryable": False,
                "timestamp": datetime.now().isoformat()
            }

            # Add details in development mode
            if self.logger.level <= logging.DEBUG:
                response["details"] = {
                    "original_error": str(error),
                    "error_type": error_type
                }

        return response

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        recent_errors = [
            err for err in self.error_history
            if (datetime.now() - datetime.fromisoformat(err["timestamp"])).seconds < 3600
        ]

        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "error_types": self.error_counts,
            "recent_error_types": {
                err_type: sum(1 for err in recent_errors if err["error_type"] == err_type)
                for err_type in set(err["error_type"] for err in recent_errors)
            }
        }

    async def handle_critical_error(self, error: Exception, context: Dict[str, Any]):
        """Handle critical errors that require immediate attention."""

        self.logger.critical(
            f"CRITICAL ERROR: {str(error)}",
            extra={
                "context": context,
                "traceback": traceback.format_exc()
            }
        )

        # Could trigger alerts, notifications, etc.
        await self._trigger_alert(error, context)

    async def _trigger_alert(self, error: Exception, context: Dict[str, Any]):
        """Trigger alert for critical errors (implement as needed)."""
        # This could send notifications to Slack, email, PagerDuty, etc.
        pass

# Integrate with Chute
@chute.on_startup()
async def initialize_error_handler(self):
    """Initialize centralized error handler."""
    self.error_handler = CentralizedErrorHandler(logger=logging.getLogger("chutes.errors"))
```

### Error Middleware

```python
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import time

class ErrorMiddleware:
    """Middleware to catch and handle all errors."""

    def __init__(self, error_handler: CentralizedErrorHandler):
        self.error_handler = error_handler

    async def __call__(self, request: Request, call_next):
        """Process request with error handling."""

        start_time = time.time()
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        try:
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as error:
            # Create error context
            context = {
                "correlation_id": correlation_id,
                "request_path": str(request.url.path),
                "request_method": request.method,
                "processing_time": time.time() - start_time,
                "user_agent": request.headers.get("User-Agent"),
                "remote_addr": request.client.host if request.client else None
            }

            # Handle error
            error_response = await self.error_handler.handle_error(error, context)

            # Determine HTTP status code
            if isinstance(error, ValidationError):
                status_code = 422
            elif isinstance(error, ModelError):
                if error.error_type == AIErrorType.OUT_OF_MEMORY:
                    status_code = 507  # Insufficient Storage
                elif error.error_type == AIErrorType.INFERENCE_TIMEOUT:
                    status_code = 504  # Gateway Timeout
                elif error.error_type == AIErrorType.INVALID_INPUT:
                    status_code = 400  # Bad Request
                else:
                    status_code = 500  # Internal Server Error
            else:
                status_code = 500

            # Create JSON response
            response = JSONResponse(
                content=error_response,
                status_code=status_code
            )
            response.headers["X-Correlation-ID"] = correlation_id

            return response

# Add middleware to Chute
@chute.on_startup()
async def add_error_middleware(self):
    """Add error handling middleware."""
    self.app.middleware("http")(ErrorMiddleware(self.error_handler))
```

## Model-Specific Error Handling

### LLM Error Handling

```python
class LLMErrorHandler:
    """Handle LLM-specific errors."""

    @staticmethod
    async def safe_generate(
        model,
        tokenizer,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Generate text with comprehensive error handling."""

        try:
            # Validate input length
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if len(inputs[0]) > model.config.max_position_embeddings:
                raise ContextLengthError(
                    len(inputs[0]),
                    model.config.max_position_embeddings
                )

            # Check available memory
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_cached = torch.cuda.memory_reserved()
                memory_total = torch.cuda.get_device_properties(0).total_memory

                if memory_allocated > memory_total * 0.9:
                    raise OutOfMemoryError(
                        memory_used=memory_allocated // (1024**2),
                        memory_available=(memory_total - memory_allocated) // (1024**2)
                    )

            # Generate with timeout
            result = await asyncio.wait_for(
                LLMErrorHandler._generate_async(model, inputs, max_tokens, temperature),
                timeout=timeout
            )

            return {
                "generated_text": result,
                "input_tokens": len(inputs[0]),
                "success": True
            }

        except asyncio.TimeoutError:
            raise InferenceTimeoutError(timeout)
        except torch.cuda.OutOfMemoryError:
            # Clear cache and retry once
            torch.cuda.empty_cache()
            try:
                result = await asyncio.wait_for(
                    LLMErrorHandler._generate_async(model, inputs, max_tokens // 2, temperature),
                    timeout=timeout
                )
                return {
                    "generated_text": result,
                    "input_tokens": len(inputs[0]),
                    "success": True,
                    "warning": "Reduced max_tokens due to memory constraints"
                }
            except:
                raise OutOfMemoryError()
        except Exception as e:
            raise ModelError(
                f"Text generation failed: {str(e)}",
                AIErrorType.GENERATION_FAILED,
                details={"original_error": str(e)},
                is_retryable=True
            )

    @staticmethod
    async def _generate_async(model, inputs, max_tokens, temperature):
        """Async wrapper for model generation."""

        def _generate():
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=model.config.eos_token_id
                )
                return model.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate)

# Usage in Chute
@chute.cord(public_api_path="/generate", method="POST")
async def generate_text_safe(self, prompt: str, max_tokens: int = 100):
    """Generate text with comprehensive error handling."""

    try:
        result = await LLMErrorHandler.safe_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens
        )
        return result

    except ModelError as e:
        # Let the middleware handle model errors
        raise
    except Exception as e:
        # Convert unexpected errors to ModelError
        raise ModelError(
            f"Unexpected error in text generation: {str(e)}",
            AIErrorType.GENERATION_FAILED,
            is_retryable=False
        )
```

### Image Generation Error Handling

```python
class ImageGenerationErrorHandler:
    """Handle image generation specific errors."""

    @staticmethod
    async def safe_generate_image(
        pipeline,
        prompt: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> Dict[str, Any]:
        """Generate image with error handling."""

        try:
            # Validate parameters
            if width * height > 1024 * 1024:
                raise ModelError(
                    "Image resolution too high",
                    AIErrorType.INVALID_INPUT,
                    details={
                        "max_resolution": "1024x1024",
                        "requested": f"{width}x{height}"
                    }
                )

            # Check memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()

            # Generate image
            image = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

            # Convert to base64
            import io
            import base64

            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

            return {
                "image": img_b64,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "success": True
            }

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            # Try with reduced parameters
            try:
                reduced_steps = max(10, num_inference_steps // 2)
                image = pipeline(
                    prompt=prompt,
                    width=min(512, width),
                    height=min(512, height),
                    num_inference_steps=reduced_steps,
                    guidance_scale=guidance_scale
                ).images[0]

                # Convert to base64
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                return {
                    "image": img_b64,
                    "width": min(512, width),
                    "height": min(512, height),
                    "steps": reduced_steps,
                    "success": True,
                    "warning": "Parameters reduced due to memory constraints"
                }
            except:
                raise OutOfMemoryError()

        except Exception as e:
            raise ModelError(
                f"Image generation failed: {str(e)}",
                AIErrorType.GENERATION_FAILED,
                details={"prompt": prompt, "parameters": {
                    "width": width, "height": height, "steps": num_inference_steps
                }},
                is_retryable=True
            )
```

## Fallback Strategies

### Model Fallback Chain

```python
class ModelFallbackChain:
    """Chain of fallback models for resilience."""

    def __init__(self):
        self.primary_model = None
        self.fallback_models = []
        self.model_health = {}

    def add_primary_model(self, model, name: str):
        """Set primary model."""
        self.primary_model = {"model": model, "name": name}
        self.model_health[name] = {"failures": 0, "last_success": datetime.now()}

    def add_fallback_model(self, model, name: str, priority: int = 1):
        """Add fallback model."""
        self.fallback_models.append({
            "model": model,
            "name": name,
            "priority": priority
        })
        self.model_health[name] = {"failures": 0, "last_success": datetime.now()}

        # Sort by priority
        self.fallback_models.sort(key=lambda x: x["priority"])

    async def generate_with_fallback(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate with automatic fallback on failure."""

        # Try primary model first
        if self.primary_model and self._is_model_healthy(self.primary_model["name"]):
            try:
                result = await self._try_model(self.primary_model, prompt, **kwargs)
                self._record_success(self.primary_model["name"])
                result["model_used"] = self.primary_model["name"]
                result["was_fallback"] = False
                return result

            except ModelError as e:
                self._record_failure(self.primary_model["name"])
                logging.warning(f"Primary model {self.primary_model['name']} failed: {e}")

        # Try fallback models
        for fallback in self.fallback_models:
            if not self._is_model_healthy(fallback["name"]):
                continue

            try:
                result = await self._try_model(fallback, prompt, **kwargs)
                self._record_success(fallback["name"])
                result["model_used"] = fallback["name"]
                result["was_fallback"] = True
                return result

            except ModelError as e:
                self._record_failure(fallback["name"])
                logging.warning(f"Fallback model {fallback['name']} failed: {e}")
                continue

        # All models failed
        raise ModelError(
            "All models in fallback chain failed",
            AIErrorType.MODEL_OVERLOADED,
            details={"tried_models": [
                self.primary_model["name"] if self.primary_model else None
            ] + [fb["name"] for fb in self.fallback_models]},
            is_retryable=True
        )

    async def _try_model(self, model_info: Dict, prompt: str, **kwargs) -> Dict[str, Any]:
        """Try generating with a specific model."""

        model = model_info["model"]

        # Implement actual model generation here
        # This is a placeholder - replace with your actual model calls
        if hasattr(model, 'generate'):
            result = await model.generate(prompt, **kwargs)
        else:
            result = f"Generated by {model_info['name']}: {prompt}"

        return {"generated_text": result}

    def _is_model_healthy(self, model_name: str) -> bool:
        """Check if model is healthy (not in circuit breaker state)."""
        health = self.model_health.get(model_name, {})

        # If too many recent failures, consider unhealthy
        if health.get("failures", 0) > 3:
            last_success = health.get("last_success", datetime.min)
            if (datetime.now() - last_success).seconds < 300:  # 5 minutes
                return False

        return True

    def _record_success(self, model_name: str):
        """Record successful model use."""
        self.model_health[model_name].update({
            "failures": 0,
            "last_success": datetime.now()
        })

    def _record_failure(self, model_name: str):
        """Record model failure."""
        self.model_health[model_name]["failures"] += 1

# Usage in Chute
@chute.on_startup()
async def initialize_fallback_chain(self):
    """Initialize model fallback chain."""
    self.fallback_chain = ModelFallbackChain()

    # Add primary model
    self.fallback_chain.add_primary_model(self.primary_llm, "gpt-3.5-turbo")

    # Add fallback models
    self.fallback_chain.add_fallback_model(self.backup_llm, "gpt-3.5-turbo-backup", priority=1)
    self.fallback_chain.add_fallback_model(self.simple_llm, "simple-model", priority=2)

@chute.cord(public_api_path="/generate_resilient", method="POST")
async def generate_with_resilience(self, prompt: str):
    """Generate text with automatic fallback."""
    return await self.fallback_chain.generate_with_fallback(prompt)
```

### Graceful Degradation

```python
class GracefulDegradationHandler:
    """Handle graceful degradation of service quality."""

    def __init__(self):
        self.degradation_levels = {
            "full": {"quality": 1.0, "speed": 1.0},
            "reduced": {"quality": 0.7, "speed": 1.5},
            "minimal": {"quality": 0.4, "speed": 3.0}
        }
        self.current_level = "full"
        self.system_load = 0.0

    def update_system_load(self, cpu_percent: float, memory_percent: float, gpu_percent: float):
        """Update system load metrics."""
        self.system_load = max(cpu_percent, memory_percent, gpu_percent) / 100.0

        # Automatically adjust degradation level
        if self.system_load > 0.9:
            self.current_level = "minimal"
        elif self.system_load > 0.7:
            self.current_level = "reduced"
        else:
            self.current_level = "full"

    def get_adjusted_parameters(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters based on current degradation level."""

        level_config = self.degradation_levels[self.current_level]
        adjusted_params = base_params.copy()

        # Adjust quality-related parameters
        if "num_inference_steps" in adjusted_params:
            adjusted_params["num_inference_steps"] = int(
                adjusted_params["num_inference_steps"] * level_config["quality"]
            )

        if "max_tokens" in adjusted_params:
            adjusted_params["max_tokens"] = int(
                adjusted_params["max_tokens"] * level_config["quality"]
            )

        # Adjust for speed (reduce batch size, etc.)
        if "batch_size" in adjusted_params:
            adjusted_params["batch_size"] = max(1, int(
                adjusted_params["batch_size"] / level_config["speed"]
            ))

        return adjusted_params

    def get_degradation_warning(self) -> Optional[str]:
        """Get warning message for current degradation level."""

        if self.current_level == "reduced":
            return "Service is running in reduced quality mode due to high system load"
        elif self.current_level == "minimal":
            return "Service is running in minimal quality mode due to very high system load"

        return None

# Usage in endpoint
@chute.cord(public_api_path="/adaptive_generate", method="POST")
async def adaptive_generate(self, prompt: str, max_tokens: int = 100):
    """Generate with adaptive quality based on system load."""

    # Get system metrics (implement based on your monitoring)
    cpu_percent = self.get_cpu_usage()
    memory_percent = self.get_memory_usage()
    gpu_percent = self.get_gpu_usage()

    # Update degradation handler
    self.degradation_handler.update_system_load(cpu_percent, memory_percent, gpu_percent)

    # Adjust parameters
    base_params = {"max_tokens": max_tokens}
    adjusted_params = self.degradation_handler.get_adjusted_parameters(base_params)

    try:
        result = await self.generate_text(prompt, **adjusted_params)

        # Add degradation warning if applicable
        warning = self.degradation_handler.get_degradation_warning()
        if warning:
            result["warning"] = warning
            result["degradation_level"] = self.degradation_handler.current_level

        return result

    except ModelError as e:
        # If still failing, try with even more conservative parameters
        if self.degradation_handler.current_level != "minimal":
            conservative_params = self.degradation_handler.get_adjusted_parameters({
                "max_tokens": max_tokens // 2
            })
            try:
                result = await self.generate_text(prompt, **conservative_params)
                result["warning"] = "Used emergency conservative parameters due to system stress"
                return result
            except:
                pass

        raise
```

## Monitoring and Alerting

### Error Metrics Collection

```python
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple

class ErrorMetricsCollector:
    """Collect and analyze error metrics."""

    def __init__(self, window_size: int = 300):  # 5 minute window
        self.window_size = window_size
        self.error_timeline = deque(maxlen=10000)  # Recent errors
        self.error_rates = defaultdict(lambda: deque(maxlen=100))
        self.error_patterns = defaultdict(int)

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None
    ):
        """Record an error occurrence."""

        timestamp = time.time()
        error_record = {
            "timestamp": timestamp,
            "error_type": error_type,
            "message": error_message,
            "context": context or {}
        }

        self.error_timeline.append(error_record)
        self.error_rates[error_type].append(timestamp)

        # Track error patterns
        pattern_key = f"{error_type}:{context.get('operation', 'unknown')}"
        self.error_patterns[pattern_key] += 1

    def get_error_rate(self, error_type: str = None, window_seconds: int = 60) -> float:
        """Get error rate (errors per minute)."""

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        if error_type:
            recent_errors = [
                t for t in self.error_rates[error_type]
                if t > cutoff_time
            ]
        else:
            recent_errors = [
                err["timestamp"] for err in self.error_timeline
                if err["timestamp"] > cutoff_time
            ]

        return len(recent_errors) * (60 / window_seconds)  # Errors per minute

    def get_top_error_patterns(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most common error patterns."""
        return sorted(
            self.error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def detect_error_spikes(self, threshold_multiplier: float = 3.0) -> List[Dict[str, Any]]:
        """Detect error rate spikes."""

        alerts = []
        current_time = time.time()

        for error_type in self.error_rates:
            # Compare recent rate to historical average
            recent_rate = self.get_error_rate(error_type, window_seconds=60)
            historical_rate = self.get_error_rate(error_type, window_seconds=3600)  # 1 hour

            if historical_rate > 0 and recent_rate > historical_rate * threshold_multiplier:
                alerts.append({
                    "type": "error_spike",
                    "error_type": error_type,
                    "recent_rate": recent_rate,
                    "historical_rate": historical_rate,
                    "multiplier": recent_rate / historical_rate,
                    "timestamp": current_time
                })

        return alerts

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""

        current_time = time.time()
        one_hour_ago = current_time - 3600

        recent_errors = [
            err for err in self.error_timeline
            if err["timestamp"] > one_hour_ago
        ]

        error_type_counts = defaultdict(int)
        for err in recent_errors:
            error_type_counts[err["error_type"]] += 1

        return {
            "total_errors_1h": len(recent_errors),
            "error_rate_1h": len(recent_errors) / 60,  # per minute
            "error_types": dict(error_type_counts),
            "top_patterns": self.get_top_error_patterns(),
            "spikes": self.detect_error_spikes()
        }

# Integrate with error handler
@chute.on_startup()
async def initialize_metrics_collector(self):
    """Initialize error metrics collection."""
    self.error_metrics = ErrorMetricsCollector()

    # Integrate with error handler
    original_handle_error = self.error_handler.handle_error

    async def handle_error_with_metrics(error, context=None, user_message=None):
        # Record metrics
        self.error_metrics.record_error(
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )

        # Call original handler
        return await original_handle_error(error, context, user_message)

    self.error_handler.handle_error = handle_error_with_metrics

@chute.cord(public_api_path="/error_metrics", method="GET")
async def get_error_metrics(self):
    """Get error metrics for monitoring."""
    return self.error_metrics.get_error_summary()
```

### Health Checks and Status

```python
class HealthChecker:
    """Comprehensive health checking for Chutes applications."""

    def __init__(self):
        self.health_checks = {}
        self.last_check_results = {}

    def register_check(self, name: str, check_func: Callable, critical: bool = False):
        """Register a health check."""
        self.health_checks[name] = {
            "func": check_func,
            "critical": critical,
            "last_result": None,
            "last_check": None
        }

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""

        results = {}
        overall_status = "healthy"
        critical_failures = []

        for name, check_info in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check_info["func"]()
                duration = time.time() - start_time

                check_result = {
                    "status": "healthy" if result.get("healthy", True) else "unhealthy",
                    "details": result,
                    "duration_ms": duration * 1000,
                    "timestamp": datetime.now().isoformat()
                }

                # Update tracking
                check_info["last_result"] = check_result
                check_info["last_check"] = time.time()

                results[name] = check_result

                # Check if this affects overall status
                if not result.get("healthy", True):
                    if check_info["critical"]:
                        overall_status = "critical"
                        critical_failures.append(name)
                    elif overall_status == "healthy":
                        overall_status = "degraded"

            except Exception as e:
                error_result = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

                results[name] = error_result

                if check_info["critical"]:
                    overall_status = "critical"
                    critical_failures.append(name)
                elif overall_status == "healthy":
                    overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "checks": results,
            "critical_failures": critical_failures,
            "timestamp": datetime.now().isoformat()
        }

    async def check_model_health(self) -> Dict[str, Any]:
        """Check model loading and basic inference."""
        try:
            # Test basic model functionality
            test_result = await self.model.generate("test", max_tokens=1)

            return {
                "healthy": True,
                "model_loaded": True,
                "inference_working": True
            }
        except Exception as e:
            return {
                "healthy": False,
                "model_loaded": hasattr(self, 'model'),
                "inference_working": False,
                "error": str(e)
            }

    async def check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU availability and memory."""
        try:
            if not torch.cuda.is_available():
                return {
                    "healthy": False,
                    "gpu_available": False,
                    "message": "CUDA not available"
                }

            device_count = torch.cuda.device_count()
            device_info = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = props.total_memory
                memory_percent = (memory_allocated / memory_total) * 100

                device_info.append({
                    "device_id": i,
                    "name": props.name,
                    "memory_used_mb": memory_allocated // (1024**2),
                    "memory_total_mb": memory_total // (1024**2),
                    "memory_percent": memory_percent
                })

            # Consider unhealthy if any GPU is over 95% memory
            gpu_healthy = all(info["memory_percent"] < 95 for info in device_info)

            return {
                "healthy": gpu_healthy,
                "gpu_available": True,
                "device_count": device_count,
                "devices": device_info
            }

        except Exception as e:
            return {
                "healthy": False,
                "gpu_available": False,
                "error": str(e)
            }

    async def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil

            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100

            return {
                "healthy": free_percent > 10,  # Unhealthy if less than 10% free
                "free_space_gb": free // (1024**3),
                "total_space_gb": total // (1024**3),
                "free_percent": free_percent
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

# Initialize health checks
@chute.on_startup()
async def initialize_health_checks(self):
    """Initialize health checking system."""
    self.health_checker = HealthChecker()

    # Register health checks
    self.health_checker.register_check("model", self.health_checker.check_model_health, critical=True)
    self.health_checker.register_check("gpu", self.health_checker.check_gpu_health, critical=True)
    self.health_checker.register_check("disk", self.health_checker.check_disk_space, critical=False)

@chute.cord(public_api_path="/health", method="GET")
async def health_check(self):
    """Health check endpoint."""
    return await self.health_checker.run_all_checks()

# Detailed status endpoint
@chute.cord(public_api_path="/status", method="GET")
async def detailed_status(self):
    """Detailed system status including errors and health."""

    health_results = await self.health_checker.run_all_checks()
    error_summary = self.error_metrics.get_error_summary()

    return {
        "health": health_results,
        "errors": error_summary,
        "uptime": time.time() - self.startup_time,
        "version": "1.0.0",  # Your app version
        "timestamp": datetime.now().isoformat()
    }
```

## Testing Error Handling

### Error Scenario Testing

```python
import pytest
from unittest.mock import Mock, patch
import asyncio

class ErrorHandlingTests:
    """Test suite for error handling scenarios."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return CentralizedErrorHandler()

    @pytest.fixture
    def mock_chute(self):
        """Create mock chute for testing."""
        chute = Mock()
        chute.error_handler = CentralizedErrorHandler()
        chute.error_metrics = ErrorMetricsCollector()
        return chute

    @pytest.mark.asyncio
    async def test_out_of_memory_handling(self, mock_chute):
        """Test OOM error handling."""

        # Simulate OOM error
        oom_error = OutOfMemoryError(memory_used=8000, memory_available=500)

        result = await mock_chute.error_handler.handle_error(oom_error)

        assert result["error_type"] == "out_of_memory"
        assert result["is_retryable"] is False
        assert "memory_used_mb" in result["details"]

    @pytest.mark.asyncio
    async def test_context_length_error(self, mock_chute):
        """Test context length error handling."""

        context_error = ContextLengthError(input_length=5000, max_length=4096)

        result = await mock_chute.error_handler.handle_error(context_error)

        assert result["error_type"] == "context_length_exceeded"
        assert "suggestion" in result["details"]
        assert result["details"]["input_length"] == 5000

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_chute):
        """Test retry with backoff."""

        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise InferenceTimeoutError(30.0)
            return "success"

        result = await failing_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mock_chute):
        """Test circuit breaker functionality."""

        call_count = 0

        @circuit_breaker(failure_threshold=2, timeout_duration=0.1)
        async def unreliable_function():
            nonlocal call_count
            call_count += 1
            raise ModelError("Simulated failure", AIErrorType.GENERATION_FAILED)

        # First two calls should fail normally
        with pytest.raises(ModelError):
            await unreliable_function()

        with pytest.raises(ModelError):
            await unreliable_function()

        # Third call should be blocked by circuit breaker
        with pytest.raises(ModelError) as exc_info:
            await unreliable_function()

        assert "Circuit breaker is OPEN" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_chain(self, mock_chute):
        """Test model fallback chain."""

        # Create mock models
        primary_model = Mock()
        primary_model.generate = Mock(side_effect=ModelError("Primary failed", AIErrorType.GENERATION_FAILED))

        fallback_model = Mock()
        fallback_model.generate = Mock(return_value="Fallback success")

        # Create fallback chain
        chain = ModelFallbackChain()
        chain.add_primary_model(primary_model, "primary")
        chain.add_fallback_model(fallback_model, "fallback")

        result = await chain.generate_with_fallback("test prompt")

        assert result["generated_text"] == "Fallback success"
        assert result["model_used"] == "fallback"
        assert result["was_fallback"] is True

    def test_error_metrics_collection(self, mock_chute):
        """Test error metrics collection."""

        metrics = ErrorMetricsCollector()

        # Record some errors
        metrics.record_error("ModelError", "Test error 1", {"operation": "inference"})
        metrics.record_error("ValidationError", "Test error 2", {"operation": "input_validation"})
        metrics.record_error("ModelError", "Test error 3", {"operation": "inference"})

        # Check metrics
        model_error_rate = metrics.get_error_rate("ModelError", window_seconds=60)
        assert model_error_rate > 0

        patterns = metrics.get_top_error_patterns()
        assert ("ModelError:inference", 2) in patterns

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_chute):
        """Test graceful degradation under load."""

        degradation_handler = GracefulDegradationHandler()

        # Simulate high load
        degradation_handler.update_system_load(cpu_percent=95, memory_percent=85, gpu_percent=90)

        assert degradation_handler.current_level == "minimal"

        # Test parameter adjustment
        base_params = {"max_tokens": 100, "num_inference_steps": 20}
        adjusted_params = degradation_handler.get_adjusted_parameters(base_params)

        assert adjusted_params["max_tokens"] < base_params["max_tokens"]
        assert adjusted_params["num_inference_steps"] < base_params["num_inference_steps"]

    @pytest.mark.asyncio
    async def test_health_checks(self, mock_chute):
        """Test health check system."""

        health_checker = HealthChecker()

        # Register mock health checks
        async def mock_healthy_check():
            return {"healthy": True, "status": "OK"}

        async def mock_unhealthy_check():
            return {"healthy": False, "status": "FAILED", "error": "Service down"}

        health_checker.register_check("service1", mock_healthy_check, critical=False)
        health_checker.register_check("service2", mock_unhealthy_check, critical=True)

        results = await health_checker.run_all_checks()

        assert results["overall_status"] == "critical"
        assert "service2" in results["critical_failures"]
        assert results["checks"]["service1"]["status"] == "healthy"
        assert results["checks"]["service2"]["status"] == "unhealthy"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Best Practices Summary

### Error Handling Checklist

```python
class ErrorHandlingBestPractices:
    """Best practices for error handling in Chutes applications."""

    CHECKLIST = {
        "Input Validation": [
            "Validate all inputs with Pydantic schemas",
            "Sanitize text inputs for security",
            "Check file uploads for type and size",
            "Provide clear validation error messages",
            "Handle edge cases (empty inputs, extreme values)"
        ],

        "Model Error Handling": [
            "Wrap model calls with appropriate try-catch blocks",
            "Handle GPU memory errors gracefully",
            "Implement timeout mechanisms for inference",
            "Check context length before processing",
            "Provide fallback models for resilience"
        ],

        "System Resilience": [
            "Implement retry mechanisms with exponential backoff",
            "Use circuit breakers to prevent cascading failures",
            "Monitor system resources and degrade gracefully",
            "Implement health checks for all critical components",
            "Log errors with sufficient context for debugging"
        ],

        "User Experience": [
            "Return user-friendly error messages",
            "Avoid exposing internal system details",
            "Provide actionable guidance in error responses",
            "Maintain consistent error response format",
            "Include correlation IDs for support requests"
        ],

        "Monitoring and Alerting": [
            "Collect comprehensive error metrics",
            "Set up alerts for error rate spikes",
            "Monitor health check failures",
            "Track error patterns and trends",
            "Implement performance degradation alerts"
        ]
    }

    @classmethod
    def validate_implementation(cls, chute_instance) -> Dict[str, bool]:
        """Validate error handling implementation."""

        results = {}

        # Check for error handler
        results["has_error_handler"] = hasattr(chute_instance, 'error_handler')

        # Check for health checks
        results["has_health_checks"] = hasattr(chute_instance, 'health_checker')

        # Check for metrics collection
        results["has_error_metrics"] = hasattr(chute_instance, 'error_metrics')

        # Check for fallback mechanisms
        results["has_fallback_chain"] = hasattr(chute_instance, 'fallback_chain')

        return results
```

## Next Steps

- **Advanced Monitoring**: Implement distributed tracing and APM integration
- **Alert Management**: Set up PagerDuty, Slack, or email alerting
- **Error Recovery**: Implement automatic recovery mechanisms
- **Performance Impact**: Minimize error handling overhead in hot paths

For more advanced topics, see:

- [Monitoring and Observability](monitoring)
- [Best Practices Guide](best-practices)
- [Production Deployment](production-deployment)
