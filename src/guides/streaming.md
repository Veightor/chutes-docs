# Real-time Streaming Responses

This guide covers how to implement real-time streaming responses in Chutes, enabling live data transmission, progressive content delivery, and interactive AI applications.

## Overview

Streaming in Chutes provides:

- **Real-time Response**: Send data as it's generated
- **Better UX**: Users see progress instead of waiting
- **Memory Efficiency**: Process large outputs without memory buildup
- **Interactive Applications**: Enable chat-like experiences
- **Scalability**: Handle long-running tasks efficiently
- **WebSocket Support**: Full duplex communication

## Basic Streaming Concepts

### HTTP Streaming vs WebSockets

```python
from chutes.chute import Chute
from fastapi import Response, WebSocket
from fastapi.responses import StreamingResponse
import asyncio
import json

chute = Chute(username="myuser", name="streaming-demo")

# HTTP Streaming - Server-sent events
@chute.cord(
    public_api_path="/stream_text",
    method="POST",
    stream=True  # Enable streaming
)
async def stream_text_generation(self, prompt: str):
    """Stream text generation token by token."""

    async def generate_tokens():
        """Generate tokens progressively."""

        # Simulate token generation
        tokens = ["Hello", " world", "!", " This", " is", " streaming", " text", "."]

        for token in tokens:
            # Yield each token as it's generated
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0.1)  # Simulate processing time

        # Send completion signal
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate_tokens(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

# WebSocket - Full duplex communication
@chute.websocket("/ws")
async def websocket_endpoint(self, websocket: WebSocket):
    """WebSocket endpoint for interactive communication."""

    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Process message
            response = await self.process_message(data)

            # Send response back
            await websocket.send_text(response)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def process_message(self, message: str) -> str:
    """Process incoming message."""
    return f"Echo: {message}"
```

## AI Model Streaming

### Streaming LLM Text Generation

```python
from typing import AsyncGenerator, Dict, Any
import time

@chute.on_startup()
async def initialize_streaming_llm(self):
    """Initialize streaming-capable LLM."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_name = "microsoft/DialoGPT-medium"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name)

    if torch.cuda.is_available():
        self.model = self.model.to("cuda")

    # Add padding token if not present
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token

async def stream_llm_generation(
    self,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream LLM generation token by token."""

    # Tokenize input
    inputs = self.tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # Generation parameters
    attention_mask = torch.ones_like(inputs)
    generated_tokens = 0

    with torch.no_grad():
        while generated_tokens < max_tokens:
            # Generate next token
            outputs = self.model(inputs, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, keepdim=True)

            # Decode token
            token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)

            # Yield token data
            yield {
                "token": token_text,
                "token_id": next_token.item(),
                "generated_tokens": generated_tokens + 1,
                "is_complete": False
            }

            # Update inputs for next iteration
            inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=attention_mask.device)], dim=-1)

            generated_tokens += 1

            # Check for end token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

    # Send completion
    yield {
        "token": "",
        "token_id": None,
        "generated_tokens": generated_tokens,
        "is_complete": True
    }

@chute.cord(
    public_api_path="/generate_stream",
    method="POST",
    stream=True
)
async def generate_streaming_text(self, prompt: str, max_tokens: int = 100):
    """Generate streaming text response."""

    async def stream_response():
        # Send SSE headers
        yield "event: start\n"
        yield f"data: {json.dumps({'message': 'Starting generation'})}\n\n"

        async for token_data in self.stream_llm_generation(prompt, max_tokens):
            if token_data["is_complete"]:
                yield "event: complete\n"
                yield f"data: {json.dumps(token_data)}\n\n"
            else:
                yield "event: token\n"
                yield f"data: {json.dumps(token_data)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )
```

### Streaming Image Generation

```python
from PIL import Image
import io
import base64

class StreamingImageGenerator:
    """Stream image generation progress."""

    def __init__(self, diffusion_model):
        self.model = diffusion_model

    async def stream_image_generation(
        self,
        prompt: str,
        steps: int = 20
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream image generation progress."""

        # Initialize generation
        yield {
            "step": 0,
            "total_steps": steps,
            "status": "initializing",
            "image": None
        }

        # Simulate diffusion steps
        for step in range(1, steps + 1):
            # Process one diffusion step
            await asyncio.sleep(0.1)  # Simulate processing

            # Every few steps, send intermediate image
            if step % 5 == 0 or step == steps:
                # Generate intermediate or final image
                if step == steps:
                    image = await self._generate_final_image(prompt)
                    status = "complete"
                else:
                    image = await self._generate_intermediate_image(prompt, step, steps)
                    status = "processing"

                # Convert image to base64
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=85)
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                yield {
                    "step": step,
                    "total_steps": steps,
                    "status": status,
                    "image": img_b64,
                    "progress": step / steps
                }
            else:
                # Send progress update without image
                yield {
                    "step": step,
                    "total_steps": steps,
                    "status": "processing",
                    "image": None,
                    "progress": step / steps
                }

    async def _generate_intermediate_image(self, prompt: str, step: int, total_steps: int):
        """Generate intermediate image (placeholder for actual implementation)."""
        # This would use your actual diffusion model's intermediate output
        # For demo, create a simple placeholder
        img = Image.new('RGB', (512, 512), color=f'#{step*10:02x}{step*5:02x}{step*15:02x}')
        return img

    async def _generate_final_image(self, prompt: str):
        """Generate final high-quality image."""
        # This would use your actual diffusion model
        img = Image.new('RGB', (512, 512), color='blue')
        return img

@chute.cord(
    public_api_path="/generate_image_stream",
    method="POST",
    stream=True
)
async def generate_streaming_image(self, prompt: str, steps: int = 20):
    """Stream image generation with progress updates."""

    generator = StreamingImageGenerator(self.diffusion_model)

    async def stream_response():
        async for update in generator.stream_image_generation(prompt, steps):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
```

## Advanced Streaming Patterns

### Chunked Data Processing

```python
from typing import AsyncIterator
import hashlib

class ChunkedProcessor:
    """Process large datasets in chunks with streaming updates."""

    async def process_large_dataset(
        self,
        data: List[str],
        chunk_size: int = 10
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process data in chunks and stream results."""

        total_items = len(data)
        processed_items = 0
        results = []

        # Process in chunks
        for i in range(0, total_items, chunk_size):
            chunk = data[i:i + chunk_size]

            # Process chunk
            chunk_results = await self._process_chunk(chunk)
            results.extend(chunk_results)
            processed_items += len(chunk)

            # Yield progress update
            yield {
                "type": "progress",
                "processed": processed_items,
                "total": total_items,
                "progress": processed_items / total_items,
                "chunk_results": chunk_results
            }

            # Allow other coroutines to run
            await asyncio.sleep(0)

        # Send final results
        yield {
            "type": "complete",
            "processed": processed_items,
            "total": total_items,
            "progress": 1.0,
            "all_results": results,
            "summary": self._generate_summary(results)
        }

    async def _process_chunk(self, chunk: List[str]) -> List[Dict[str, Any]]:
        """Process a single chunk of data."""
        results = []

        for item in chunk:
            # Simulate processing
            await asyncio.sleep(0.01)

            result = {
                "original": item,
                "processed": item.upper(),
                "length": len(item),
                "hash": hashlib.md5(item.encode()).hexdigest()[:8]
            }
            results.append(result)

        return results

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_length = sum(r["length"] for r in results)
        avg_length = total_length / len(results) if results else 0

        return {
            "total_items": len(results),
            "total_length": total_length,
            "average_length": avg_length
        }

@chute.cord(
    public_api_path="/process_stream",
    method="POST",
    stream=True
)
async def process_data_stream(self, data: List[str], chunk_size: int = 10):
    """Stream large data processing."""

    processor = ChunkedProcessor()

    async def stream_response():
        async for update in processor.process_large_dataset(data, chunk_size):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
```

### Multi-Model Streaming Pipeline

```python
class StreamingPipeline:
    """Stream processing through multiple AI models."""

    def __init__(self):
        self.models = {}

    async def stream_multi_model_processing(
        self,
        text: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process text through multiple models with streaming updates."""

        pipeline_steps = [
            ("preprocessing", self._preprocess),
            ("sentiment", self._analyze_sentiment),
            ("entities", self._extract_entities),
            ("summary", self._generate_summary),
            ("translation", self._translate_text)
        ]

        current_data = {"text": text}

        for step_name, step_func in pipeline_steps:
            yield {
                "step": step_name,
                "status": "starting",
                "input": current_data
            }

            try:
                # Process step
                step_result = await step_func(current_data)
                current_data.update(step_result)

                yield {
                    "step": step_name,
                    "status": "completed",
                    "result": step_result,
                    "accumulated_data": current_data
                }

            except Exception as e:
                yield {
                    "step": step_name,
                    "status": "error",
                    "error": str(e)
                }
                break

        # Send final result
        yield {
            "step": "pipeline_complete",
            "status": "completed",
            "final_result": current_data
        }

    async def _preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocessing step."""
        await asyncio.sleep(0.1)
        return {
            "cleaned_text": data["text"].strip().lower(),
            "word_count": len(data["text"].split())
        }

    async def _analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sentiment analysis step."""
        await asyncio.sleep(0.2)
        # Simulate sentiment analysis
        return {
            "sentiment": "positive",
            "sentiment_score": 0.8
        }

    async def _extract_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Entity extraction step."""
        await asyncio.sleep(0.15)
        return {
            "entities": [
                {"text": "example", "type": "MISC", "confidence": 0.9}
            ]
        }

    async def _generate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Text summarization step."""
        await asyncio.sleep(0.3)
        return {
            "summary": f"Summary of: {data['text'][:50]}..."
        }

    async def _translate_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translation step."""
        await asyncio.sleep(0.25)
        return {
            "translated_text": f"Translated: {data['text']}"
        }

@chute.cord(
    public_api_path="/pipeline_stream",
    method="POST",
    stream=True
)
async def stream_pipeline_processing(self, text: str):
    """Stream multi-model pipeline processing."""

    pipeline = StreamingPipeline()

    async def stream_response():
        async for update in pipeline.stream_multi_model_processing(text):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )
```

## WebSocket Applications

### Interactive Chat Application

```python
from typing import Dict, Set
import uuid

class ChatManager:
    """Manage WebSocket chat sessions."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.chat_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str = None):
        """Connect a new WebSocket client."""
        await websocket.accept()

        if session_id is None:
            session_id = str(uuid.uuid4())

        self.active_connections[session_id] = websocket
        self.chat_sessions[session_id] = {
            "messages": [],
            "connected_at": time.time()
        }

        # Send welcome message
        await self.send_message(session_id, {
            "type": "system",
            "message": f"Connected to chat session {session_id}",
            "session_id": session_id
        })

        return session_id

    async def disconnect(self, session_id: str):
        """Disconnect a WebSocket client."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]

    async def send_message(self, session_id: str, message: Dict):
        """Send message to specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message))

    async def broadcast_message(self, message: Dict, exclude_session: str = None):
        """Broadcast message to all connected sessions."""
        for session_id, websocket in self.active_connections.items():
            if session_id != exclude_session:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    # Connection may be closed
                    pass

@chute.on_startup()
async def initialize_chat(self):
    """Initialize chat manager."""
    self.chat_manager = ChatManager()

@chute.websocket("/chat")
async def chat_websocket(self, websocket: WebSocket, session_id: str = None):
    """WebSocket endpoint for interactive chat."""

    session_id = await self.chat_manager.connect(websocket, session_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Process based on message type
            if message_data.get("type") == "user_message":
                await self._handle_user_message(session_id, message_data)
            elif message_data.get("type") == "typing":
                await self._handle_typing_indicator(session_id, message_data)
            elif message_data.get("type") == "ping":
                await self._handle_ping(session_id)

    except Exception as e:
        print(f"Chat error for session {session_id}: {e}")
    finally:
        await self.chat_manager.disconnect(session_id)

async def _handle_user_message(self, session_id: str, message_data: Dict):
    """Handle user message and generate AI response."""

    user_message = message_data.get("message", "")

    # Store user message
    self.chat_manager.chat_sessions[session_id]["messages"].append({
        "role": "user",
        "content": user_message,
        "timestamp": time.time()
    })

    # Send typing indicator
    await self.chat_manager.send_message(session_id, {
        "type": "ai_typing",
        "typing": True
    })

    # Generate streaming AI response
    ai_response = ""
    async for token_data in self.stream_llm_generation(user_message):
        if not token_data["is_complete"]:
            ai_response += token_data["token"]

            # Send partial response
            await self.chat_manager.send_message(session_id, {
                "type": "ai_message_partial",
                "content": ai_response,
                "token": token_data["token"]
            })
        else:
            # Send complete response
            await self.chat_manager.send_message(session_id, {
                "type": "ai_message_complete",
                "content": ai_response
            })

            # Store AI message
            self.chat_manager.chat_sessions[session_id]["messages"].append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": time.time()
            })

async def _handle_typing_indicator(self, session_id: str, message_data: Dict):
    """Handle typing indicator."""
    typing = message_data.get("typing", False)

    # Broadcast typing status to other users (if multi-user chat)
    await self.chat_manager.broadcast_message({
        "type": "user_typing",
        "session_id": session_id,
        "typing": typing
    }, exclude_session=session_id)

async def _handle_ping(self, session_id: str):
    """Handle ping for connection keepalive."""
    await self.chat_manager.send_message(session_id, {
        "type": "pong",
        "timestamp": time.time()
    })
```

### Real-time Collaboration

```python
class CollaborativeEditor:
    """Real-time collaborative document editing."""

    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.subscribers: Dict[str, Set[str]] = {}  # doc_id -> set of session_ids
        self.session_connections: Dict[str, WebSocket] = {}

    async def join_document(self, doc_id: str, session_id: str, websocket: WebSocket):
        """Join a collaborative document."""

        # Initialize document if doesn't exist
        if doc_id not in self.documents:
            self.documents[doc_id] = {
                "content": "",
                "version": 0,
                "last_modified": time.time()
            }
            self.subscribers[doc_id] = set()

        # Add subscriber
        self.subscribers[doc_id].add(session_id)
        self.session_connections[session_id] = websocket

        # Send current document state
        await websocket.send_text(json.dumps({
            "type": "document_state",
            "doc_id": doc_id,
            "content": self.documents[doc_id]["content"],
            "version": self.documents[doc_id]["version"]
        }))

        # Notify other users
        await self._broadcast_to_document(doc_id, {
            "type": "user_joined",
            "session_id": session_id
        }, exclude_session=session_id)

    async def leave_document(self, doc_id: str, session_id: str):
        """Leave a collaborative document."""
        if doc_id in self.subscribers:
            self.subscribers[doc_id].discard(session_id)

        if session_id in self.session_connections:
            del self.session_connections[session_id]

        # Notify other users
        await self._broadcast_to_document(doc_id, {
            "type": "user_left",
            "session_id": session_id
        }, exclude_session=session_id)

    async def apply_operation(self, doc_id: str, session_id: str, operation: Dict):
        """Apply an edit operation to the document."""

        if doc_id not in self.documents:
            return

        doc = self.documents[doc_id]

        # Apply operation (simplified - real implementation would use OT)
        if operation["type"] == "insert":
            pos = operation["position"]
            text = operation["text"]
            content = doc["content"]
            doc["content"] = content[:pos] + text + content[pos:]

        elif operation["type"] == "delete":
            start = operation["start"]
            length = operation["length"]
            content = doc["content"]
            doc["content"] = content[:start] + content[start + length:]

        # Update version
        doc["version"] += 1
        doc["last_modified"] = time.time()

        # Broadcast operation to other users
        await self._broadcast_to_document(doc_id, {
            "type": "operation",
            "operation": operation,
            "version": doc["version"],
            "author": session_id
        }, exclude_session=session_id)

    async def _broadcast_to_document(self, doc_id: str, message: Dict, exclude_session: str = None):
        """Broadcast message to all document subscribers."""
        if doc_id not in self.subscribers:
            return

        for session_id in self.subscribers[doc_id]:
            if session_id != exclude_session and session_id in self.session_connections:
                try:
                    websocket = self.session_connections[session_id]
                    await websocket.send_text(json.dumps(message))
                except:
                    # Connection may be closed
                    pass

@chute.websocket("/collaborate/{doc_id}")
async def collaborative_editing(self, websocket: WebSocket, doc_id: str):
    """WebSocket endpoint for collaborative editing."""

    session_id = str(uuid.uuid4())
    editor = getattr(self, 'collaborative_editor', None)

    if editor is None:
        self.collaborative_editor = CollaborativeEditor()
        editor = self.collaborative_editor

    await websocket.accept()
    await editor.join_document(doc_id, session_id, websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "operation":
                await editor.apply_operation(doc_id, session_id, message["operation"])
            elif message["type"] == "cursor_position":
                # Broadcast cursor position to other users
                await editor._broadcast_to_document(doc_id, {
                    "type": "cursor_update",
                    "session_id": session_id,
                    "position": message["position"]
                }, exclude_session=session_id)

    except Exception as e:
        print(f"Collaboration error: {e}")
    finally:
        await editor.leave_document(doc_id, session_id)
```

## Performance and Optimization

### Streaming Buffer Management

```python
import asyncio
from collections import deque

class StreamingBuffer:
    """Manage streaming data with buffering and backpressure handling."""

    def __init__(self, max_buffer_size: int = 1000):
        self.buffer = deque(maxlen=max_buffer_size)
        self.consumers = set()
        self.producer_task = None
        self.is_producing = False

    async def start_producing(self, producer_func):
        """Start producing data."""
        if self.is_producing:
            return

        self.is_producing = True
        self.producer_task = asyncio.create_task(self._produce_data(producer_func))

    async def stop_producing(self):
        """Stop producing data."""
        self.is_producing = False
        if self.producer_task:
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                pass

    async def _produce_data(self, producer_func):
        """Internal producer loop."""
        try:
            async for data in producer_func():
                self.buffer.append(data)

                # Notify consumers
                await self._notify_consumers(data)

                # Backpressure handling
                if len(self.buffer) >= self.buffer.maxlen * 0.8:
                    await asyncio.sleep(0.01)  # Slow down production

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Producer error: {e}")
        finally:
            self.is_producing = False

    async def _notify_consumers(self, data):
        """Notify all consumers of new data."""
        dead_consumers = set()

        for consumer in self.consumers:
            try:
                await consumer.put_nowait(data)
            except:
                dead_consumers.add(consumer)

        # Remove dead consumers
        self.consumers -= dead_consumers

    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to the stream."""
        consumer_queue = asyncio.Queue(maxsize=100)
        self.consumers.add(consumer_queue)

        # Send buffered data to new consumer
        for data in self.buffer:
            await consumer_queue.put(data)

        return consumer_queue

    def unsubscribe(self, consumer_queue: asyncio.Queue):
        """Unsubscribe from the stream."""
        self.consumers.discard(consumer_queue)

# Usage in streaming endpoint
@chute.on_startup()
async def initialize_streaming_buffer(self):
    """Initialize streaming buffer."""
    self.streaming_buffer = StreamingBuffer(max_buffer_size=500)

@chute.cord(
    public_api_path="/buffered_stream",
    method="GET",
    stream=True
)
async def buffered_streaming_endpoint(self):
    """Stream with buffering and backpressure handling."""

    # Start producing if not already started
    if not self.streaming_buffer.is_producing:
        await self.streaming_buffer.start_producing(self._data_producer)

    # Subscribe to stream
    consumer_queue = await self.streaming_buffer.subscribe()

    async def stream_response():
        try:
            while True:
                # Get data from buffer
                data = await asyncio.wait_for(consumer_queue.get(), timeout=30.0)
                yield f"data: {json.dumps(data)}\n\n"

        except asyncio.TimeoutError:
            yield "event: timeout\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            self.streaming_buffer.unsubscribe(consumer_queue)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )

async def _data_producer(self):
    """Example data producer."""
    counter = 0
    while True:
        yield {
            "timestamp": time.time(),
            "counter": counter,
            "data": f"Generated data {counter}"
        }
        counter += 1
        await asyncio.sleep(0.1)
```

### Connection Management

```python
class ConnectionManager:
    """Manage WebSocket connections with health monitoring."""

    def __init__(self):
        self.connections: Dict[str, Dict] = {}
        self.monitoring_task = None

    async def start_monitoring(self):
        """Start connection health monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_connections())

    async def stop_monitoring(self):
        """Stop connection monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

    async def add_connection(self, session_id: str, websocket: WebSocket):
        """Add a new WebSocket connection."""
        self.connections[session_id] = {
            "websocket": websocket,
            "connected_at": time.time(),
            "last_ping": time.time(),
            "is_alive": True
        }

        # Start monitoring if first connection
        if len(self.connections) == 1:
            await self.start_monitoring()

    async def remove_connection(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.connections:
            del self.connections[session_id]

        # Stop monitoring if no connections
        if len(self.connections) == 0:
            await self.stop_monitoring()

    async def send_to_connection(self, session_id: str, message: Dict) -> bool:
        """Send message to specific connection."""
        if session_id not in self.connections:
            return False

        try:
            websocket = self.connections[session_id]["websocket"]
            await websocket.send_text(json.dumps(message))
            return True
        except:
            # Mark connection as dead
            self.connections[session_id]["is_alive"] = False
            return False

    async def broadcast(self, message: Dict, exclude: Set[str] = None):
        """Broadcast message to all connections."""
        if exclude is None:
            exclude = set()

        dead_connections = []

        for session_id, conn_info in self.connections.items():
            if session_id not in exclude and conn_info["is_alive"]:
                success = await self.send_to_connection(session_id, message)
                if not success:
                    dead_connections.append(session_id)

        # Clean up dead connections
        for session_id in dead_connections:
            await self.remove_connection(session_id)

    async def _monitor_connections(self):
        """Monitor connection health."""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_time = time.time()
                dead_connections = []

                for session_id, conn_info in self.connections.items():
                    # Check if connection is stale
                    if current_time - conn_info["last_ping"] > 60:  # 1 minute timeout
                        dead_connections.append(session_id)
                        continue

                    # Send ping
                    success = await self.send_to_connection(session_id, {
                        "type": "ping",
                        "timestamp": current_time
                    })

                    if success:
                        conn_info["last_ping"] = current_time
                    else:
                        dead_connections.append(session_id)

                # Clean up dead connections
                for session_id in dead_connections:
                    await self.remove_connection(session_id)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Connection monitoring error: {e}")
```

## Client-Side Integration

### JavaScript/TypeScript Client

```javascript
class ChutesStreamingClient {
	constructor(baseUrl) {
		this.baseUrl = baseUrl;
		this.eventSource = null;
		this.websocket = null;
	}

	// HTTP Streaming (Server-Sent Events)
	streamHTTP(endpoint, options = {}) {
		return new Promise((resolve, reject) => {
			const url = `${this.baseUrl}${endpoint}`;

			this.eventSource = new EventSource(url);

			const results = [];

			this.eventSource.onmessage = (event) => {
				try {
					const data = JSON.parse(event.data);
					results.push(data);

					// Call progress callback if provided
					if (options.onProgress) {
						options.onProgress(data);
					}

					// Check for completion
					if (data.done || data.is_complete) {
						this.eventSource.close();
						resolve(results);
					}
				} catch (e) {
					console.error('Failed to parse SSE data:', e);
				}
			};

			this.eventSource.onerror = (error) => {
				this.eventSource.close();
				reject(error);
			};
		});
	}

	// WebSocket Streaming
	async connectWebSocket(endpoint) {
		return new Promise((resolve, reject) => {
			const wsUrl = `ws${
				this.baseUrl.startsWith('https') ? 's' : ''
			}://${this.baseUrl.replace(/^https?:\/\//, '')}${endpoint}`;

			this.websocket = new WebSocket(wsUrl);

			this.websocket.onopen = () => {
				resolve(this);
			};

			this.websocket.onerror = (error) => {
				reject(error);
			};

			this.websocket.onclose = () => {
				console.log('WebSocket connection closed');
			};
		});
	}

	// Send message via WebSocket
	sendMessage(message) {
		if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
			this.websocket.send(JSON.stringify(message));
		}
	}

	// Set message handler
	onMessage(handler) {
		if (this.websocket) {
			this.websocket.onmessage = (event) => {
				try {
					const data = JSON.parse(event.data);
					handler(data);
				} catch (e) {
					console.error('Failed to parse WebSocket message:', e);
				}
			};
		}
	}

	// Clean up connections
	disconnect() {
		if (this.eventSource) {
			this.eventSource.close();
			this.eventSource = null;
		}

		if (this.websocket) {
			this.websocket.close();
			this.websocket = null;
		}
	}
}

// Usage examples
const client = new ChutesStreamingClient('https://myuser-my-chute.chutes.ai');

// HTTP Streaming example
client
	.streamHTTP('/generate_stream', {
		onProgress: (data) => {
			console.log('Received token:', data.token);
			// Update UI with streaming content
			document.getElementById('output').textContent += data.token;
		}
	})
	.then((results) => {
		console.log('Streaming complete:', results);
	});

// WebSocket example
client.connectWebSocket('/chat').then(() => {
	client.onMessage((data) => {
		if (data.type === 'ai_message_partial') {
			// Update chat interface with partial message
			updateChatInterface(data.content);
		}
	});

	// Send a message
	client.sendMessage({
		type: 'user_message',
		message: 'Hello, AI!'
	});
});
```

### Python Client

```python
import asyncio
import aiohttp
import json
from typing import AsyncIterator, Callable, Optional

class ChutesAsyncClient:
    """Async Python client for Chutes streaming APIs."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def stream_http(
        self,
        endpoint: str,
        method: str = 'GET',
        data: dict = None,
        progress_callback: Callable = None
    ) -> AsyncIterator[dict]:
        """Stream data via HTTP Server-Sent Events."""

        url = f"{self.base_url}{endpoint}"

        async with self.session.request(
            method,
            url,
            json=data,
            headers={'Accept': 'text/event-stream'}
        ) as response:

            async for line in response.content:
                line_str = line.decode('utf-8').strip()

                if line_str.startswith('data: '):
                    try:
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        data_obj = json.loads(data_str)

                        if progress_callback:
                            progress_callback(data_obj)

                        yield data_obj

                    except json.JSONDecodeError:
                        continue

    async def connect_websocket(
        self,
        endpoint: str,
        message_handler: Callable = None
    ):
        """Connect to WebSocket endpoint."""

        ws_url = f"ws{self.base_url[4:]}{endpoint}"

        async with self.session.ws_connect(ws_url) as ws:
            self.websocket = ws

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if message_handler:
                            await message_handler(data)
                        yield data
                    except json.JSONDecodeError:
                        continue
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    async def send_websocket_message(self, message: dict):
        """Send message via WebSocket."""
        if hasattr(self, 'websocket'):
            await self.websocket.send_str(json.dumps(message))

# Usage example
async def example_usage():
    async with ChutesAsyncClient('https://myuser-my-chute.chutes.ai') as client:

        # HTTP Streaming
        async for token_data in client.stream_http(
            '/generate_stream',
            method='POST',
            data={'prompt': 'Tell me a story'},
            progress_callback=lambda x: print(f"Token: {x.get('token', '')}")
        ):
            if token_data.get('is_complete'):
                print("Generation complete!")
                break

        # WebSocket example
        async for message in client.connect_websocket(
            '/chat',
            message_handler=lambda msg: print(f"Received: {msg}")
        ):
            if message.get('type') == 'system':
                # Send a message
                await client.send_websocket_message({
                    'type': 'user_message',
                    'message': 'Hello from Python client!'
                })

# Run the example
# asyncio.run(example_usage())
```

## Best Practices and Troubleshooting

### Error Handling in Streams

```python
class StreamErrorHandler:
    """Handle errors in streaming applications."""

    @staticmethod
    async def safe_stream_wrapper(stream_func, error_callback=None):
        """Wrap streaming function with error handling."""

        try:
            async for item in stream_func():
                yield item
        except asyncio.CancelledError:
            yield {"type": "error", "error": "Stream cancelled"}
        except Exception as e:
            error_msg = {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

            if error_callback:
                await error_callback(error_msg)

            yield error_msg

    @staticmethod
    async def retry_stream(stream_func, max_retries=3, delay=1.0):
        """Retry streaming function on failure."""

        for attempt in range(max_retries):
            try:
                async for item in stream_func():
                    yield item
                return  # Success, exit retry loop

            except Exception as e:
                if attempt == max_retries - 1:
                    yield {
                        "type": "error",
                        "error": f"Failed after {max_retries} attempts: {str(e)}"
                    }
                    return

                yield {
                    "type": "retry",
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": str(e)
                }

                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff

# Usage
@chute.cord(public_api_path="/safe_stream", method="POST", stream=True)
async def safe_streaming_endpoint(self, prompt: str):
    """Streaming endpoint with error handling."""

    async def stream_with_errors():
        error_handler = StreamErrorHandler()

        async for item in error_handler.safe_stream_wrapper(
            lambda: self.stream_llm_generation(prompt),
            error_callback=lambda err: self.log_error(err)
        ):
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        stream_with_errors(),
        media_type="text/event-stream"
    )
```

### Performance Monitoring

```python
class StreamingMetrics:
    """Monitor streaming performance."""

    def __init__(self):
        self.active_streams = 0
        self.total_streams = 0
        self.avg_stream_duration = 0
        self.stream_start_times = {}

    def start_stream(self, stream_id: str):
        """Record stream start."""
        self.active_streams += 1
        self.total_streams += 1
        self.stream_start_times[stream_id] = time.time()

    def end_stream(self, stream_id: str):
        """Record stream end."""
        self.active_streams = max(0, self.active_streams - 1)

        if stream_id in self.stream_start_times:
            duration = time.time() - self.stream_start_times[stream_id]
            self.avg_stream_duration = (
                (self.avg_stream_duration * (self.total_streams - 1) + duration) /
                self.total_streams
            )
            del self.stream_start_times[stream_id]

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return {
            "active_streams": self.active_streams,
            "total_streams": self.total_streams,
            "avg_duration": self.avg_stream_duration,
            "current_streams": list(self.stream_start_times.keys())
        }

@chute.on_startup()
async def initialize_metrics(self):
    """Initialize streaming metrics."""
    self.streaming_metrics = StreamingMetrics()

@chute.cord(public_api_path="/metrics", method="GET")
async def get_streaming_metrics(self):
    """Get streaming performance metrics."""
    return self.streaming_metrics.get_metrics()
```

## Next Steps

- **Advanced Protocols**: Implement WebRTC for peer-to-peer streaming
- **Scale Optimization**: Handle thousands of concurrent streams
- **Security**: Implement authentication and rate limiting for streams
- **Integration**: Connect with real-time databases and message queues

For more advanced topics, see:

- [Error Handling Guide](error-handling)
- [Best Practices](best-practices)
- [Performance Optimization](performance-optimization)
