# Running a Chute

This guide demonstrates how to call and run chutes in your applications using various programming languages. We'll cover examples for Python, TypeScript, Go, and Rust.

## Overview

Chutes can be invoked via simple HTTP POST requests to the endpoint:

```
POST https://{username}-{chute-name}.chutes.ai/{path}
```

Or using the API endpoint:

```
POST https://api.chutes.ai/chutes/{chute-id}/{path}
```

## Authentication

All requests require authentication using either:
- API Key in the `X-API-Key` header
- Bearer token in the `Authorization` header

## Python Example (using aiohttp)

### Basic LLM Invocation

```python
import aiohttp
import asyncio
import json

async def call_llm_chute():
    url = "https://myuser-my-llm.chutes.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key-here"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            print(result["choices"][0]["message"]["content"])

# Run the async function
asyncio.run(call_llm_chute())
```

### Streaming Response

```python
import aiohttp
import asyncio
import json

async def stream_llm_response():
    url = "https://myuser-my-llm.chutes.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key-here"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Write a short story about AI"}
        ],
        "stream": True,
        "max_tokens": 500
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        if data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                content = chunk["choices"][0]["delta"].get("content", "")
                                print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                pass

asyncio.run(stream_llm_response())
```

### Image Generation

```python
import aiohttp
import asyncio
import base64

async def generate_image():
    url = "https://myuser-my-diffusion.chutes.ai/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key-here"
    }
    
    payload = {
        "prompt": "A beautiful sunset over mountains, oil painting style",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            
            # Save the image
            image_data = base64.b64decode(result["data"][0]["b64_json"])
            with open("generated_image.png", "wb") as f:
                f.write(image_data)
            print("Image saved as generated_image.png")

asyncio.run(generate_image())
```

## TypeScript Example

### Basic LLM Invocation

```typescript
async function callLLMChute() {
    const url = "https://myuser-my-llm.chutes.ai/v1/chat/completions";
    
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": "your-api-key-here"
        },
        body: JSON.stringify({
            model: "meta-llama/Llama-3.1-8B-Instruct",
            messages: [
                { role: "system", content: "You are a helpful assistant." },
                { role: "user", content: "Hello! How are you?" }
            ],
            max_tokens: 100,
            temperature: 0.7
        })
    });
    
    const result = await response.json();
    console.log(result.choices[0].message.content);
}

callLLMChute();
```

### Streaming Response

```typescript
async function streamLLMResponse() {
    const url = "https://myuser-my-llm.chutes.ai/v1/chat/completions";
    
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": "your-api-key-here"
        },
        body: JSON.stringify({
            model: "meta-llama/Llama-3.1-8B-Instruct",
            messages: [
                { role: "user", content: "Write a short story about AI" }
            ],
            stream: true,
            max_tokens: 500
        })
    });
    
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data !== '[DONE]') {
                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices[0].delta?.content || '';
                        process.stdout.write(content);
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
        }
    }
}

streamLLMResponse();
```

### Image Generation

```typescript
import * as fs from 'fs';

async function generateImage() {
    const url = "https://myuser-my-diffusion.chutes.ai/v1/images/generations";
    
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": "your-api-key-here"
        },
        body: JSON.stringify({
            prompt: "A beautiful sunset over mountains, oil painting style",
            n: 1,
            size: "1024x1024",
            response_format: "b64_json"
        })
    });
    
    const result = await response.json();
    
    // Save the image
    const imageData = Buffer.from(result.data[0].b64_json, 'base64');
    fs.writeFileSync('generated_image.png', imageData);
    console.log('Image saved as generated_image.png');
}

generateImage();
```

## Go Example

### Basic LLM Invocation

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type ChatRequest struct {
    Model       string    `json:"model"`
    Messages    []Message `json:"messages"`
    MaxTokens   int       `json:"max_tokens"`
    Temperature float64   `json:"temperature"`
}

type ChatResponse struct {
    Choices []struct {
        Message struct {
            Content string `json:"content"`
        } `json:"message"`
    } `json:"choices"`
}

func callLLMChute() error {
    url := "https://myuser-my-llm.chutes.ai/v1/chat/completions"
    
    request := ChatRequest{
        Model: "meta-llama/Llama-3.1-8B-Instruct",
        Messages: []Message{
            {Role: "system", Content: "You are a helpful assistant."},
            {Role: "user", Content: "Hello! How are you?"},
        },
        MaxTokens:   100,
        Temperature: 0.7,
    }
    
    jsonData, err := json.Marshal(request)
    if err != nil {
        return err
    }
    
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
        return err
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-API-Key", "your-api-key-here")
    
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return err
    }
    
    var response ChatResponse
    err = json.Unmarshal(body, &response)
    if err != nil {
        return err
    }
    
    fmt.Println(response.Choices[0].Message.Content)
    return nil
}

func main() {
    if err := callLLMChute(); err != nil {
        fmt.Printf("Error: %v\n", err)
    }
}
```

### Streaming Response

```go
package main

import (
    "bufio"
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
)

type StreamRequest struct {
    Model     string    `json:"model"`
    Messages  []Message `json:"messages"`
    Stream    bool      `json:"stream"`
    MaxTokens int       `json:"max_tokens"`
}

func streamLLMResponse() error {
    url := "https://myuser-my-llm.chutes.ai/v1/chat/completions"
    
    request := StreamRequest{
        Model: "meta-llama/Llama-3.1-8B-Instruct",
        Messages: []Message{
            {Role: "user", Content: "Write a short story about AI"},
        },
        Stream:    true,
        MaxTokens: 500,
    }
    
    jsonData, err := json.Marshal(request)
    if err != nil {
        return err
    }
    
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
        return err
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-API-Key", "your-api-key-here")
    
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    scanner := bufio.NewScanner(resp.Body)
    for scanner.Scan() {
        line := scanner.Text()
        if strings.HasPrefix(line, "data: ") {
            data := line[6:]
            if data != "[DONE]" {
                var chunk map[string]interface{}
                if err := json.Unmarshal([]byte(data), &chunk); err == nil {
                    if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
                        if choice, ok := choices[0].(map[string]interface{}); ok {
                            if delta, ok := choice["delta"].(map[string]interface{}); ok {
                                if content, ok := delta["content"].(string); ok {
                                    fmt.Print(content)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return scanner.Err()
}

func main() {
    if err := streamLLMResponse(); err != nil {
        fmt.Printf("Error: %v\n", err)
    }
}
```

## Rust Example

### Basic LLM Invocation

```rust
use reqwest;
use serde::{Deserialize, Serialize};
use tokio;

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: i32,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageResponse,
}

#[derive(Deserialize)]
struct MessageResponse {
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://myuser-my-llm.chutes.ai/v1/chat/completions";
    
    let request = ChatRequest {
        model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Hello! How are you?".to_string(),
            },
        ],
        max_tokens: 100,
        temperature: 0.7,
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("X-API-Key", "your-api-key-here")
        .json(&request)
        .send()
        .await?
        .json::<ChatResponse>()
        .await?;
    
    if let Some(choice) = response.choices.first() {
        println!("{}", choice.message.content);
    }
    
    Ok(())
}
```

### Streaming Response

```rust
use futures_util::StreamExt;
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json;
use tokio;

#[derive(Serialize)]
struct StreamRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    max_tokens: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://myuser-my-llm.chutes.ai/v1/chat/completions";
    
    let request = StreamRequest {
        model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: "Write a short story about AI".to_string(),
            },
        ],
        stream: true,
        max_tokens: 500,
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("X-API-Key", "your-api-key-here")
        .json(&request)
        .send()
        .await?;
    
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].to_string();
            buffer = buffer[pos + 1..].to_string();
            
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data != "[DONE]" {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                            print!("{}", content);
                            use std::io::{self, Write};
                            io::stdout().flush().unwrap();
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}
```

### Image Generation (Rust)

```rust
use base64;
use reqwest;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use tokio;

#[derive(Serialize)]
struct ImageRequest {
    prompt: String,
    n: i32,
    size: String,
    response_format: String,
}

#[derive(Deserialize)]
struct ImageResponse {
    data: Vec<ImageData>,
}

#[derive(Deserialize)]
struct ImageData {
    b64_json: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://myuser-my-diffusion.chutes.ai/v1/images/generations";
    
    let request = ImageRequest {
        prompt: "A beautiful sunset over mountains, oil painting style".to_string(),
        n: 1,
        size: "1024x1024".to_string(),
        response_format: "b64_json".to_string(),
    };
    
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("X-API-Key", "your-api-key-here")
        .json(&request)
        .send()
        .await?
        .json::<ImageResponse>()
        .await?;
    
    if let Some(image_data) = response.data.first() {
        let decoded = base64::decode(&image_data.b64_json)?;
        let mut file = File::create("generated_image.png")?;
        file.write_all(&decoded)?;
        println!("Image saved as generated_image.png");
    }
    
    Ok(())
}
```

## Error Handling

All examples should include proper error handling. Common error codes:

- `401`: Invalid or missing API key
- `403`: Access denied to the chute
- `404`: Chute not found
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Service temporarily unavailable

Example error handling in Python:

```python
async def call_with_error_handling():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error = await response.text()
                    print(f"Error {response.status}: {error}")
                    return None
    except aiohttp.ClientError as e:
        print(f"Request failed: {e}")
        return None
```

## Best Practices

1. **Use Environment Variables**: Store API keys in environment variables rather than hardcoding them
2. **Implement Retries**: Add retry logic for transient failures
3. **Handle Rate Limits**: Respect rate limits and implement backoff strategies
4. **Stream Large Responses**: Use streaming for long-form content generation
5. **Set Timeouts**: Configure appropriate timeouts for your use case
6. **Monitor Usage**: Track API usage to manage costs effectively

## Next Steps

- Learn about [Authentication](/docs/getting-started/authentication)
- Explore [Templates](/docs/templates) for specific use cases
- Check the [API Reference](/docs/api) for detailed endpoint documentation
- See [Examples](/docs/examples) for more complex implementations