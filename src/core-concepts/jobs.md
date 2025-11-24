# Jobs (Background Tasks)

**Jobs** are background tasks in Chutes that handle long-running operations, file uploads, and asynchronous processing. Unlike Cords (API endpoints), Jobs don't need to respond immediately and can run for extended periods.

## What is a Job?

A Job is a decorated function that can:

- 🕐 **Run for extended periods** (hours or days)
- 📁 **Handle file uploads** and downloads
- 🔄 **Process data asynchronously**
- 💾 **Store results** in persistent storage
- 📊 **Track progress** and status
- 🔄 **Retry on failure** automatically

## Basic Job Definition

```python
from chutes.chute import Chute

chute = Chute(username="myuser", name="my-chute", image="my-image")

@chute.job(timeout=3600)  # 1 hour timeout
async def process_data(self, data: dict) -> dict:
    # Long-running processing logic
    result = await expensive_computation(data)
    return {"status": "completed", "result": result}
```

## Job Decorator Parameters

### `timeout: int = 300`

Maximum time the job can run (in seconds).

```python
@chute.job(timeout=7200)  # 2 hours
async def long_training_job(self, config: dict):
    # Training logic that might take hours
    pass
```

### `upload: bool = False`

Whether the job accepts file uploads.

```python
@chute.job(upload=True, timeout=1800)
async def process_video(self, video_file: bytes) -> dict:
    # Process uploaded video file
    return {"processed": True}
```

### `retry: int = 0`

Number of automatic retries on failure.

```python
@chute.job(retry=3, timeout=600)
async def unreliable_task(self, data: dict):
    # Will retry up to 3 times if it fails
    pass
```

## Input Types

### Simple Data

```python
@chute.job()
async def analyze_text(self, text: str, language: str = "en") -> dict:
    analysis = await perform_analysis(text, language)
    return {"sentiment": analysis.sentiment, "topics": analysis.topics}
```

### Structured Input with Pydantic

```python
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    model_type: str
    learning_rate: float
    epochs: int
    batch_size: int

@chute.job(timeout=14400)  # 4 hours
async def train_model(self, config: TrainingConfig) -> dict:
    model = create_model(config.model_type)
    results = await train(model, config)
    return {"accuracy": results.accuracy, "loss": results.final_loss}
```

### File Uploads

```python
@chute.job(upload=True, timeout=3600)
async def process_dataset(self, dataset_file: bytes) -> dict:
    # Save uploaded file
    with open("/tmp/dataset.csv", "wb") as f:
        f.write(dataset_file)

    # Process the dataset
    df = pd.read_csv("/tmp/dataset.csv")
    results = analyze_dataset(df)

    return {"rows": len(df), "analysis": results}
```

## Progress Tracking

For long-running jobs, you can track and report progress:

```python
@chute.job(timeout=7200)
async def batch_process(self, items: list) -> dict:
    results = []
    total = len(items)

    for i, item in enumerate(items):
        # Process each item
        result = await process_item(item)
        results.append(result)

        # Report progress (this is logged)
        progress = (i + 1) / total * 100
        print(f"Progress: {progress:.1f}% ({i+1}/{total})")

    return {"processed": len(results), "results": results}
```

## Error Handling

```python
@chute.job(retry=2, timeout=1800)
async def resilient_job(self, data: dict) -> dict:
    try:
        result = await risky_operation(data)
        return {"success": True, "result": result}
    except TemporaryError as e:
        # This will trigger a retry
        raise e
    except PermanentError as e:
        # Return error instead of raising to avoid retries
        return {"success": False, "error": str(e)}
```

## Working with Files

### Processing Uploaded Files

```python
import tempfile
import os

@chute.job(upload=True, timeout=1800)
async def process_image(self, image_file: bytes) -> dict:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file)
        tmp_path = tmp.name

    try:
        # Process the image
        processed = await image_processing_function(tmp_path)
        return {"processed": True, "features": processed}
    finally:
        # Clean up
        os.unlink(tmp_path)
```

### Generating Files for Download

```python
@chute.job(timeout=3600)
async def generate_report(self, report_config: dict) -> dict:
    # Generate report
    report_data = await create_report(report_config)

    # Save to file (this could be uploaded to cloud storage)
    report_path = f"/tmp/report_{report_config['id']}.pdf"
    save_report_as_pdf(report_data, report_path)

    return {
        "report_generated": True,
        "report_path": report_path,
        "pages": len(report_data)
    }
```

## State Management

Jobs can maintain state throughout their execution:

```python
@chute.job(timeout=7200)
async def training_job(self, config: dict) -> dict:
    # Initialize training state
    self.training_state = {
        "epoch": 0,
        "best_accuracy": 0.0,
        "model_checkpoints": []
    }

    for epoch in range(config["epochs"]):
        self.training_state["epoch"] = epoch

        # Train for one epoch
        accuracy = await train_epoch(epoch)

        if accuracy > self.training_state["best_accuracy"]:
            self.training_state["best_accuracy"] = accuracy
            # Save checkpoint
            checkpoint_path = f"/tmp/checkpoint_epoch_{epoch}.pt"
            save_checkpoint(checkpoint_path)
            self.training_state["model_checkpoints"].append(checkpoint_path)

    return self.training_state
```

## Job Lifecycle

1. **Queued**: Job is submitted and waiting to run
2. **Running**: Job is executing
3. **Completed**: Job finished successfully
4. **Failed**: Job encountered an error
5. **Retrying**: Job failed but will retry (if retry > 0)
6. **Timeout**: Job exceeded timeout limit

## Running Jobs

### Programmatically

```python
# Submit a job
job_id = await chute.submit_job("process_data", {"input": "data"})

# Check job status
status = await chute.get_job_status(job_id)

# Get job results (when completed)
results = await chute.get_job_results(job_id)
```

### Via HTTP API

```bash
# Submit a job
curl -X POST https://your-username-your-chute.chutes.ai/jobs/process_data \
  -H "Content-Type: application/json" \
  -d '{"input": "data"}'

# Check status
curl https://your-username-your-chute.chutes.ai/jobs/{job_id}/status

# Get results
curl https://your-username-your-chute.chutes.ai/jobs/{job_id}/results
```

## Best Practices

### 1. Set Appropriate Timeouts

```python
# Short tasks
@chute.job(timeout=300)  # 5 minutes

# Medium tasks
@chute.job(timeout=1800)  # 30 minutes

# Long training jobs
@chute.job(timeout=14400)  # 4 hours
```

### 2. Handle Failures Gracefully

```python
@chute.job(retry=2)
async def robust_job(self, data: dict) -> dict:
    try:
        return await process_data(data)
    except Exception as e:
        # Log the error
        logger.error(f"Job failed: {e}")
        # Return error info instead of raising
        return {"success": False, "error": str(e)}
```

### 3. Use Progress Tracking

```python
@chute.job(timeout=3600)
async def batch_job(self, items: list) -> dict:
    for i, item in enumerate(items):
        # Process item
        await process_item(item)

        # Log progress every 10 items
        if i % 10 == 0:
            print(f"Processed {i}/{len(items)} items")
```

### 4. Clean Up Resources

```python
@chute.job(timeout=1800)
async def file_processing_job(self, data: dict) -> dict:
    temp_files = []
    try:
        # Create temporary files
        for file_data in data["files"]:
            tmp_file = create_temp_file(file_data)
            temp_files.append(tmp_file)

        # Process files
        results = await process_files(temp_files)
        return results
    finally:
        # Always clean up
        for tmp_file in temp_files:
            os.unlink(tmp_file)
```

## Common Use Cases

### Model Training

```python
@chute.job(timeout=14400, retry=1)
async def train_custom_model(self, training_data: dict) -> dict:
    # Load training data
    dataset = load_dataset(training_data["dataset_path"])

    # Initialize model
    model = create_model(training_data["model_config"])

    # Train model
    for epoch in range(training_data["epochs"]):
        loss = await train_epoch(model, dataset)
        print(f"Epoch {epoch}: Loss = {loss}")

    # Save trained model
    model_path = f"/tmp/trained_model_{int(time.time())}.pt"
    save_model(model, model_path)

    return {"model_path": model_path, "final_loss": loss}
```

### Data Processing Pipeline

```python
@chute.job(upload=True, timeout=7200)
async def process_pipeline(self, raw_data: bytes) -> dict:
    # Stage 1: Parse data
    parsed_data = parse_raw_data(raw_data)
    print(f"Parsed {len(parsed_data)} records")

    # Stage 2: Clean data
    cleaned_data = clean_data(parsed_data)
    print(f"Cleaned data, {len(cleaned_data)} records remaining")

    # Stage 3: Transform data
    transformed_data = transform_data(cleaned_data)
    print(f"Transformed data complete")

    # Stage 4: Generate insights
    insights = generate_insights(transformed_data)

    return {
        "records_processed": len(parsed_data),
        "records_final": len(transformed_data),
        "insights": insights
    }
```

### Batch Image Processing

```python
@chute.job(timeout=3600)
async def batch_image_process(self, image_urls: list) -> dict:
    results = []

    for i, url in enumerate(image_urls):
        try:
            # Download and process image
            image = await download_image(url)
            processed = await process_image(image)
            results.append({"url": url, "success": True, "result": processed})
        except Exception as e:
            results.append({"url": url, "success": False, "error": str(e)})

        # Progress update
        if i % 10 == 0:
            print(f"Processed {i}/{len(image_urls)} images")

    success_count = sum(1 for r in results if r["success"])
    return {
        "total": len(image_urls),
        "successful": success_count,
        "failed": len(image_urls) - success_count,
        "results": results
    }
```

## Next Steps

- **[Chutes](/docs/core-concepts/chutes)** - Learn about the main Chute class
- **[Cords](/docs/core-concepts/cords)** - Understand API endpoints
- **[Images](/docs/core-concepts/images)** - Build custom Docker environments
- **[Your First Custom Chute](/docs/getting-started/first-chute)** - Complete example walkthrough
