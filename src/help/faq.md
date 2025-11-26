# Frequently Asked Questions (FAQ)

Common questions and answers about Chutes SDK and platform.

## General Questions

### What is Chutes?

Chutes is a serverless AI compute platform that lets you deploy and scale AI models on GPU infrastructure without managing servers. You write Python code using our SDK, and we handle the infrastructure, scaling, and deployment.

**Key benefits:**

- Deploy AI models in minutes, not hours
- Pay only for actual compute time used
- Automatic scaling from 0 to hundreds of instances
- Access to latest GPU hardware (H200, Mi300x, B200, etc.)
- No DevOps or Kubernetes knowledge required

### How is Chutes different from other platforms?

| Feature        | Chutes          | Traditional Cloud | Other AI Platforms |
| -------------- | --------------- | ----------------- | ------------------ |
| **Setup Time** | Minutes         | Hours/Days        | Hours              |
| **Scaling**    | Automatic (0→∞) | Manual            | Limited            |
| **Pricing**    | Pay-per-use     | Always-on         | Subscription       |
| **GPU Access** | Latest hardware | Limited selection | Restricted         |
| **Code Style** | Simple Python   | Complex configs   | Platform-specific  |

### Who should use Chutes?

**Perfect for:**

- AI/ML engineers building production applications
- Startups needing scalable AI infrastructure
- Researchers requiring powerful GPU compute
- Companies wanting serverless AI deployment

**Use cases:**

- LLM chat applications
- Image/video generation services
- Real-time AI APIs
- Batch processing workflows
- Model inference at scale

### Is Chutes suitable for production?

Yes! Chutes is designed for production workloads with:

- 99.9% uptime SLA
- Enterprise security and compliance
- Confidential Compute with Trusted Execution Environments (TEE)
- Global edge deployment
- Automatic failover and recovery
- 24/7 monitoring and support

## Getting Started

### How do I get started with Chutes?

1. **Install the SDK**

   ```bash
   pip install chutes
   ```

2. **Create account and authenticate**

   ```bash
   chutes auth login
   ```

3. **Deploy your first chute**

   ```python
   from chutes.chute import Chute

   chute = Chute(username="myuser", name="hello-world")

   @chute.cord(public_api_path="/hello")
   async def hello():
       return {"message": "Hello, World!"}
   ```

   ```bash
   chutes deploy
   ```

### Do I need Docker experience?

No! Chutes handles containerization automatically. However, if you need custom dependencies, you can optionally use our `Image` class:

```python
from chutes.image import Image

# Simple dependency installation
image = Image(
    base_image="nvidia/cuda:12.1-runtime-ubuntu22.04"
).run([
    "pip install transformers torch"
])

chute = Chute(
    username="myuser",
    name="my-app",
    image=image
)
```

### What programming languages are supported?

Currently, Chutes supports **Python only**. We're considering other languages based on user demand.

**Python versions supported:**

- Python 3.8+
- Recommended: Python 3.10 or 3.11

### Can I use my existing Python code?

Yes! Chutes is designed to work with existing Python codebases. You typically just need to:

1. Wrap your functions with `@chute.cord` decorators
2. Add any dependencies to an `Image` if needed
3. Deploy with `chutes deploy`

## Deployment & Usage

### How long does deployment take?

- **First deployment**: 5-15 minutes (includes image building)
- **Code-only updates**: 1-3 minutes
- **No-code config updates**: 30 seconds

### Can I deploy multiple versions?

Yes! Each deployment creates a new version:

```bash
# Deploy new version
chutes deploy

# List versions
chutes chutes versions <chute-name>

# Rollback to previous version
chutes chutes rollback <chute-name> --version v1.2.3
```

### How does scaling work?

Chutes automatically scales based on traffic:

- **Scale to zero**: No requests = no costs
- **Auto-scaling**: Handles traffic spikes automatically
- **Global load balancing**: Requests routed to optimal regions
- **Cold start optimization**: Fast instance startup

```python
# Configure scaling behavior
chute = Chute(
    username="myuser",
    name="my-app",
    min_replicas=0,    # Scale to zero
    max_replicas=100   # Scale up to 100 instances
)
```

### Can I deploy the same model multiple times?

Yes! You can have multiple deployments:

```python
# Production deployment
prod_chute = Chute(
    username="myuser",
    name="llm-prod",
    node_selector=NodeSelector()
)

# Development deployment
dev_chute = Chute(
    username="myuser",
    name="llm-dev",
    node_selector=NodeSelector()
)
```

### How do I handle different environments?

Use environment variables and different chute names:

```python
import os

environment = os.getenv("ENVIRONMENT", "dev")
chute_name = f"my-app-{environment}"

chute = Chute(username="myuser", name=chute_name)
```

## Performance & Optimization

### How can I optimize performance?

**Model optimization:**

```python
# Use optimized engines
from chutes.templates import build_vllm_chute

chute = build_vllm_chute(
    username="myuser",
    name="fast-llm",
    model_name="microsoft/DialoGPT-medium",
    engine_args={
        "gpu_memory_utilization": 0.9,
        "enable_chunked_prefill": True,
        "use_v2_block_manager": True
    }
)
```

**Hardware selection:**

```python
# Choose appropriate hardware
from chutes.chute import NodeSelector

node_selector = NodeSelector(
    gpu_count=1,
    min_vram_gb_per_gpu=24,
    include=["h100", "a100"]  # High-performance GPUs
)
```

**Caching strategies:**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(input_hash):
    return compute_result(input_hash)
```

### What's the latency for API calls?

Typical latencies:

- **Warm instances**: 50-200ms
- **Cold start**: 5-30 seconds (depending on model size)
- **Global edge**: <100ms additional routing overhead

### How do I minimize cold starts?

```python
# Keep minimum replicas warm
chute = Chute(
    username="myuser",
    name="low-latency-app",
    min_replicas=1  # Always keep 1 instance warm
)
```

```python
# Optimize startup time
@chute.on_startup()
async def setup(self):
    # Load models efficiently
    self.model = load_model_optimized()
```

### Can I use multiple GPUs?

Yes! Specify multiple GPUs in your node selector:

```python
# Multi-GPU setup
node_selector = NodeSelector(
    gpu_count=4,  # Use 4 GPUs
    min_vram_gb_per_gpu=40
)

# Distribute model across GPUs
@chute.on_startup()
async def setup(self):
    self.model = load_model_distributed(device_map="auto")
```

## Pricing & Billing

### How does pricing work?

Chutes uses **pay-per-use** pricing:

- **Compute**: Per GPU-second of actual usage

### How can I control costs?

**Use spot instances:**

```python
node_selector = NodeSelector()
```

**Scale to zero:**

```python
chute = Chute(
    username="myuser",
    name="cost-optimized",
    min_replicas=0  # No idle costs
)
```

**Choose appropriate hardware:**

```python
# Cost-effective GPUs for development
node_selector = NodeSelector(
    include=["l40", "a6000"],  # Less expensive than H100
    exclude=["h100", "h200"]
)
```

**Monitor usage:**

```bash
# Check current usage
chutes account usage

# Set billing alerts
chutes account alerts --threshold 100
```

### Do you offer volume discounts?

Yes! We offer:

- **Startup credits**: Up to $10,000 for qualifying startups
- **Enterprise pricing**: Custom rates for large usage
- **Volume discounts**: Automatic discounts at usage tiers

Fill in [this form](https://forms.rayonlabs.ai/chutes-sales) to contact sales.

## Features & Capabilities

### What AI frameworks are supported?

**Officially supported:**

- **PyTorch**: Full support with CUDA optimization
- **Transformers**: Hugging Face models and pipelines
- **VLLM**: High-performance LLM inference
- **SGLang**: Structured generation for LLMs
- **Diffusers**: Image/video generation models

**Community supported:**

- TensorFlow/Keras
- JAX/Flax
- ONNX Runtime
- OpenCV
- scikit-learn

### Can I use custom models?

Absolutely! Upload your models several ways:

```python
# From Hugging Face Hub
model_name = "your-username/custom-model"

# From local files
image = Image().copy("./my-model/", "/opt/model/")

# From cloud storage
image = Image().run([
    "wget https://storage.example.com/model.bin -O /opt/model.bin"
])
```

### Do you support streaming responses?

Yes! Perfect for LLM chat applications:

```python
from typing import AsyncGenerator

@chute.cord(public_api_path="/stream")
async def stream_generate(self, prompt: str) -> AsyncGenerator[str, None]:
    async for token in self.model.stream_generate(prompt):
        yield f"data: {token}\n\n"
```

### Can I run background jobs?

Yes! Use the `@chute.job` decorator:

```python
@chute.job()
async def process_batch(self, batch_data: List[str]):
    results = []
    for item in batch_data:
        result = await self.process_item(item)
        results.append(result)
    return results

# Trigger job
@chute.cord(public_api_path="/submit_batch")
async def submit_batch(self, data: List[str]):
    job_id = await self.process_batch(data)
    return {"job_id": job_id}
```

### Is there a Python client library?

Yes! Use the generated client or standard HTTP:

```python
# Generated client (coming soon)
from chutes.client import ChuteClient

client = ChuteClient("https://your-chute.chutes.ai")
result = await client.predict(text="Hello world")

# Standard HTTP requests
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://your-chute.chutes.ai/predict",
        json={"text": "Hello world"}
    )
    result = response.json()
```

## Technical Details

### What regions are available?

**Current regions:**

- **US**: us-west-2 (Oregon), us-east-1 (Virginia)
- **Europe**: eu-west-1 (Ireland), eu-central-1 (Frankfurt)
- **Asia**: ap-southeast-1 (Singapore), ap-northeast-1 (Tokyo)

**Coming soon:**

- us-central-1, eu-west-2, ap-south-1

### What GPU types are available?

| GPU       | VRAM      | Best For                 | Pricing Tier |
| --------- | --------- | ------------------------ | ------------ |
| **T4**    | 16GB      | Small models, dev        | $            |
| **V100**  | 16GB/32GB | Training, medium models  | $$           |
| **A6000** | 48GB      | Production inference     | $$$          |
| **L40**   | 48GB      | Cost-effective inference | $$$          |
| **A100**  | 40GB/80GB | Large models, training   | $$$$         |
| **H100**  | 80GB      | Latest generation        | $$$$$        |
| **H200**  | 141GB     | Massive models           | $$$$$        |

### How does networking work?

- **Public endpoints**: HTTPS with automatic SSL certificates
- **Private endpoints**: VPC peering for enterprise customers
- **Load balancing**: Automatic traffic distribution
- **CDN**: Global content delivery for static assets

### What about data persistence?

**Temporary storage** (included):

- Container filesystem
- Cleared on restart/redeploy

**Persistent storage** (optional):

```python
chute = Chute(
    username="myuser",
    name="persistent-app",
    storage_gb=100  # 100GB persistent disk
)

# Access at /opt/storage/
@chute.cord(public_api_path="/save")
async def save_data(self, data: str):
    with open("/opt/storage/data.txt", "w") as f:
        f.write(data)
```

### Can I access the underlying infrastructure?

Chutes is serverless, so direct infrastructure access isn't available. However, you get:

- **System info**: CPU, memory, GPU details via APIs
- **Metrics**: Performance monitoring and alerts
- **Logs**: Comprehensive application and system logs
- **Debug endpoints**: Custom debugging interfaces

## Troubleshooting

### My deployment is failing. What should I check?

1. **Validate configuration:**

   ```bash
   chutes chutes validate --file chute.py
   ```

2. **Check build logs:**

   ```bash
   chutes chutes logs --build-logs <chute-name>
   ```

3. **Verify resource availability:**

   ```bash
   chutes nodes list --available
   ```

4. **Common fixes:**
   - Reduce GPU requirements
   - Enable spot instances
   - Use more flexible node selector
   - Check dependency versions

### I'm getting out of memory errors. How do I fix this?

**Immediate fixes:**

```python
# Request more VRAM
node_selector = NodeSelector(min_vram_gb_per_gpu=48)

# Or reduce batch size
engine_args = {"max_num_batched_tokens": 1024}

# Enable memory optimization
engine_args = {"gpu_memory_utilization": 0.85}
```

See the [Troubleshooting Guide](/docs/help/troubleshooting) for more details.

### How do I debug performance issues?

```python
# Add performance monitoring
import time

@chute.cord(public_api_path="/predict")
async def predict(self, input_data):
    start_time = time.time()
    result = await self.model.predict(input_data)
    duration = time.time() - start_time
    self.logger.info(f"Prediction took {duration:.2f}s")
    return result

# Check resource usage
@chute.cord(public_api_path="/stats")
async def get_stats(self):
    return {
        "gpu_memory": torch.cuda.memory_allocated(),
        "cpu_percent": psutil.cpu_percent()
    }
```

## Integrations

### Can I integrate with my existing CI/CD?

Yes! Chutes works with any CI/CD system:

**GitHub Actions:**

```yaml
name: Deploy to Chutes
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install Chutes
        run: pip install chutes
      - name: Deploy
        run: chutes deploy --name my-app-prod
        env:
          CHUTES_API_KEY: ${{ secrets.CHUTES_API_KEY }}
```

### Does it work with monitoring tools?

Yes! Export metrics to your preferred tools:

```python
# Prometheus metrics
@chute.cord(public_api_path="/metrics")
async def metrics(self):
    return generate_prometheus_metrics()

# Custom webhooks
@chute.cord(public_api_path="/predict")
async def predict(self, input_data):
    result = await self.model.predict(input_data)

    # Send to monitoring
    await send_to_datadog(metric="prediction_count", value=1)

    return result
```

### Can I use it with databases?

Absolutely! Connect to any database:

```python
# PostgreSQL example
import asyncpg

@chute.on_startup()
async def setup(self):
    self.db = await asyncpg.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

@chute.cord(public_api_path="/query")
async def query_data(self, query: str):
    rows = await self.db.fetch("SELECT * FROM table WHERE condition = $1", query)
    return [dict(row) for row in rows]
```

## Security & Privacy

### How secure is my data?

**Infrastructure security:**

- SOC 2 Type II compliance
- End-to-end encryption (TLS 1.3)
- Network isolation between deployments
- Regular security audits and penetration testing

**Data handling:**

- No persistent storage of request/response data
- Optional data encryption at rest
- GDPR and CCPA compliant
- Customer data never used for training

### Can I use private models?

Yes! Several options for private models:

```python
# Private Hugging Face models (requires token)
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token"

# Upload during build
image = Image().copy("./private-model/", "/opt/model/")

# Download from private S3
image = Image().run([
    "aws s3 cp s3://private-bucket/model.bin /opt/model.bin"
]).env("AWS_ACCESS_KEY_ID", "your_key")
```

## Still have questions?

- **Community**: Join our [Discord](https://discord.gg/wHrXwWkCRz) for community support
- **Documentation**: Check our [comprehensive docs](/docs)
- **Support**: Email `support@chutes.ai` for technical assistance
- **Sales**: Fill in this form: [Form](https://forms.rayonlabs.ai/chutes-sales)

We're constantly updating this FAQ based on user feedback. If you have a question not covered here, please let us know!
