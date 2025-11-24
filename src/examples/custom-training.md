# Custom Model Training with Chutes

This guide demonstrates how to train custom machine learning models using Chutes, from data preparation through deployment of the trained models.

## Overview

Custom training enables:

- **Fine-tuning Pre-trained Models**: Adapt existing models to your specific use case
- **Training from Scratch**: Build models for unique domains or tasks
- **Distributed Training**: Scale training across multiple GPUs and nodes
- **Experiment Tracking**: Monitor training progress and compare experiments
- **Model Versioning**: Manage different model versions and deployments

## Quick Start

### Basic Fine-tuning Setup

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TrainingConfig(BaseModel):
    model_name: str
    dataset_path: str
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    output_dir: str = "/models/output"
    save_steps: int = 500
    eval_steps: int = 100

# Training image with ML frameworks
training_image = (
    Image(
        username="myuser",
        name="custom-training",
        tag="1.0.0",
        base_image="nvidia/cuda:12.1-devel-ubuntu22.04",
        python_version="3.11"
    )
    .run_command("pip install torch==2.1.0+cu121 transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0 wandb==0.16.0 tensorboard==2.15.0 --extra-index-url https://download.pytorch.org/whl/cu121")
    .add("./training", "/app/training")
    .add("./data", "/app/data")
)
```

## Text Classification Fine-tuning

### Complete Training Pipeline

```python
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

class TextClassificationTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize W&B for experiment tracking
        wandb.init(
            project="chutes-training",
            config=config.dict(),
            name=f"training-{config.model_name.replace('/', '-')}"
        )

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with number of labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.get_label_names())
        )

        # Resize token embeddings if necessary
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_and_prepare_data(self):
        """Load and preprocess training data"""
        self.logger.info(f"Loading dataset from: {self.config.dataset_path}")

        # Load dataset (assumes CSV format with 'text' and 'label' columns)
        if self.config.dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=self.config.dataset_path)['train']
        else:
            dataset = load_dataset(self.config.dataset_path)['train']

        # Split into train/validation
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

        # Tokenize datasets
        self.train_dataset = dataset['train'].map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        self.val_dataset = dataset['test'].map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset['test'].column_names
        )

        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")

    def tokenize_function(self, examples):
        """Tokenize text data"""
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Will be handled by data collator
            max_length=512
        )

        # Convert labels to integers if they're strings
        if isinstance(examples['label'][0], str):
            label_names = self.get_label_names()
            label_to_id = {name: idx for idx, name in enumerate(label_names)}
            tokenized['labels'] = [label_to_id[label] for label in examples['label']]
        else:
            tokenized['labels'] = examples['label']

        return tokenized

    def get_label_names(self):
        """Get unique label names from dataset"""
        # This should be implemented based on your specific dataset
        # For example, for sentiment analysis:
        return ["negative", "neutral", "positive"]

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self):
        """Train the model"""
        self.logger.info("Starting training...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            warmup_steps=100,
            fp16=True,  # Enable mixed precision training
            dataloader_num_workers=4,
            report_to="wandb"
        )

        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        train_result = trainer.train()

        # Save the final model
        trainer.save_model()
        trainer.save_state()

        # Log final metrics
        self.logger.info(f"Training completed!")
        self.logger.info(f"Final train loss: {train_result.training_loss}")

        # Final evaluation
        eval_result = trainer.evaluate()
        self.logger.info(f"Final evaluation: {eval_result}")

        return trainer

async def run_training(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Main training entry point"""
    config = TrainingConfig(**inputs['config'])

    trainer = TextClassificationTrainer(config)

    # Load model and data
    trainer.load_model_and_tokenizer()
    trainer.load_and_prepare_data()

    # Train the model
    trained_model = trainer.train()

    return {
        "status": "completed",
        "model_path": config.output_dir,
        "training_samples": len(trainer.train_dataset),
        "validation_samples": len(trainer.val_dataset)
    }
```

### Deploy Training Chute

```python
# Create training chute
training_chute = Chute(
    username="myuser",
    name="text-classification-training",
    image=training_image,
    entry_file="training.py",
    entry_point="run_training",
    node_selector=NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=24),
    timeout_seconds=3600,  # 1 hour for training
    concurrency=1  # Training should run sequentially
)

# Start training
training_config = {
    "config": {
        "model_name": "bert-base-uncased",
        "dataset_path": "/app/data/sentiment_dataset.csv",
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "output_dir": "/models/sentiment-classifier"
    }
}

result = training_chute.run(training_config)
print(f"Training result: {result}")
```

## Computer Vision Training

### Image Classification

```python
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import timm
from PIL import Image

class ImageClassificationTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def load_model(self, num_classes: int):
        """Load pre-trained vision model"""
        if "vit" in self.config.model_name.lower():
            # Vision Transformer
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=True,
                num_classes=num_classes
            )
        else:
            # ResNet or other CNN
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.model.to(self.device)

    def prepare_data(self):
        """Prepare image datasets"""
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Load datasets
        train_dataset = datasets.ImageFolder(
            root=f"{self.config.dataset_path}/train",
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=f"{self.config.dataset_path}/val",
            transform=val_transform
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return len(train_dataset.classes)

    def train(self):
        """Train the vision model"""
        num_classes = self.prepare_data()
        self.load_model(num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs
        )

        best_val_acc = 0.0

        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            # Validation phase
            val_acc = self.evaluate()
            scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(),
                          f"{self.config.output_dir}/best_model.pth")

            print(f'Epoch {epoch}: Train Acc: {100.*train_correct/train_total:.2f}%, '
                  f'Val Acc: {val_acc:.2f}%')

    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100. * correct / total
```

## Distributed Training

### Multi-GPU Training Setup

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class DistributedTrainer:
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config

        # Initialize distributed training
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')

    def setup_model(self, model):
        """Setup model for distributed training"""
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.rank])
        return model

    def setup_dataloader(self, dataset, batch_size):
        """Setup distributed dataloader"""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        return dataloader, sampler

    def train_epoch(self, model, dataloader, optimizer, criterion, epoch):
        """Train one epoch with distributed setup"""
        model.train()
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if self.rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        return total_loss / len(dataloader)

def run_distributed_training(rank, world_size, config):
    """Run distributed training on multiple GPUs"""
    trainer = DistributedTrainer(rank, world_size, config)

    # Setup model, data, etc.
    # ... (model and data setup code)

    # Cleanup
    dist.destroy_process_group()

async def run_multi_gpu_training(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Launch multi-GPU training"""
    config = TrainingConfig(**inputs['config'])
    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(
            run_distributed_training,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        trainer = TextClassificationTrainer(config)
        trainer.train()

    return {"status": "completed", "gpus_used": world_size}
```

## Model Deployment Pipeline

### Trained Model Serving

```python
from chutes.chute import Chute
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ModelInferenceService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on input text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        }

# Global model instance
model_service = None

async def load_model(model_path: str):
    """Load trained model for inference"""
    global model_service
    model_service = ModelInferenceService(model_path)
    return {"status": "model_loaded"}

async def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Inference endpoint"""
    text = inputs["text"]
    result = model_service.predict(text)
    return result

# Deploy inference service
inference_chute = Chute(
    username="myuser",
    name="trained-model-inference",
    image=training_image,  # Reuse training image
    entry_file="inference.py",
    entry_point="predict",
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=8
    ),
    timeout_seconds=60,
    concurrency=10
)
```

## Experiment Tracking

### Advanced Monitoring

```python
import mlflow
import mlflow.pytorch
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, model_name: str):
        """Log trained model"""
        mlflow.pytorch.log_model(model, model_name)

    def log_artifacts(self, local_path: str):
        """Log training artifacts"""
        mlflow.log_artifacts(local_path)

    def finish(self):
        """End experiment run"""
        mlflow.end_run()

# Integration with training
class TrackedTrainer(TextClassificationTrainer):
    def __init__(self, config: TrainingConfig, experiment_name: str):
        super().__init__(config)
        self.tracker = ExperimentTracker(experiment_name)

        # Log hyperparameters
        self.tracker.log_params(config.dict())

    def train(self):
        """Training with experiment tracking"""
        trainer = super().train()

        # Log final model
        self.tracker.log_model(self.model, "final_model")
        self.tracker.log_artifacts(self.config.output_dir)
        self.tracker.finish()

        return trainer
```

## Next Steps

- **[Model Deployment](../guides/model-deployment)** - Deploy trained models at scale
- **[Performance Optimization](../guides/performance)** - Optimize training performance
- **[MLOps Pipelines](../guides/mlops)** - Production ML workflows
- **[Advanced Training](../guides/advanced-training)** - Advanced training techniques

For production training workflows, see the [Enterprise Training Guide](../guides/enterprise-training).
