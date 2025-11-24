# Deploying Chutes

The `chutes deploy` command takes your built images and deploys them as live, scalable AI applications on the Chutes platform.

## Basic Deploy Command

### `chutes deploy`

Deploy a chute to the platform.

```bash
chutes deploy <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image
- `--debug`: Enable debug logging
- `--public`: Mark chute as public
- `--tag TEXT`: Specific image tag to deploy
- `--wait`: Wait for deployment to complete
- `--env-file TEXT`: Environment variables file
- `--scale INTEGER`: Initial number of instances

## Deployment Examples

### Basic Deployment

```bash
# Deploy latest built image
chutes deploy my_chute:chute
```

**What happens:**

- ✅ Validates image exists
- ✅ Creates deployment configuration
- ✅ Provisions GPU resources
- ✅ Starts your chute
- ✅ Returns public URL

### Production Deployment

```bash
# Deploy specific version with logo
chutes deploy my_chute:chute \
  --tag v1.2.0 \
  --logo ./assets/logo.png \
  --public \
  --wait
```

### Deployment with Environment Variables

```bash
# Create environment file
cat > .env << EOF
MODEL_PATH=/app/models/custom
DEBUG_MODE=false
API_TIMEOUT=30
EOF

# Deploy with environment
chutes deploy my_chute:chute --env-file .env
```

### Scaled Deployment

```bash
# Deploy with multiple instances
chutes deploy my_chute:chute --scale 3 --wait
```

## Deployment Process

### Deployment Stages

```bash
# Example deployment output
Deploying chute: my_chute:chute
✓ Validating image
✓ Creating deployment
✓ Provisioning resources
✓ Starting instances
✓ Health checks passing
✓ Configuring networking
✓ Enabling auto-scaling

Deployment successful!
🌐 URL: https://myuser-my-chute.chutes.ai
📋 Chute ID: 12345678-1234-5678-9abc-123456789012
📊 Status: Running (1/1 instances)
```

### Resource Allocation

During deployment, Chutes:

1. **Analyzes Requirements**: Reads your `NodeSelector` configuration
2. **Finds Available Hardware**: Matches your requirements to available GPUs
3. **Provisions Resources**: Allocates GPU, CPU, and memory
4. **Network Setup**: Creates load balancer and SSL certificates
5. **Health Monitoring**: Sets up health checks and monitoring

## Advanced Deployment Options

### Environment Configuration

```bash
# Method 1: Environment file
cat > production.env << EOF
# Model configuration
MODEL_NAME=microsoft/DialoGPT-medium
MODEL_REVISION=main
BATCH_SIZE=16

# Performance tuning
GPU_MEMORY_FRACTION=0.9
ENABLE_MIXED_PRECISION=true

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
EOF

chutes deploy my_chute:chute --env-file production.env
```

```bash
# Method 2: Direct environment variables
export MODEL_NAME=microsoft/DialoGPT-medium
export BATCH_SIZE=16
chutes deploy my_chute:chute
```

### Public vs Private Deployments

```bash
# Private deployment (default)
chutes deploy my_chute:chute

# Public deployment (visible in marketplace)
chutes deploy my_chute:chute --public
```

**Public Deployment Benefits:**

- 📈 Marketplace visibility
- 👥 Community discovery
- 💰 Potential revenue sharing
- 🏆 Featured deployment opportunities

### Blue-Green Deployments

```bash
# Deploy new version alongside existing
chutes deploy my_chute:chute --tag v2.0.0 --alias green

# Test new version
curl https://myuser-my-chute-green.chutes.ai/health

# Switch traffic to new version
chutes deployments switch my_chute green

# Remove old version
chutes deployments remove my_chute blue
```

## Deployment Strategies

### Rolling Updates

```bash
# Update existing deployment
chutes deploy my_chute:chute --tag v1.1.0 --strategy rolling

# Monitor rollout
chutes deployments status my_chute --watch
```

### Canary Deployments

```bash
# Deploy canary with 10% traffic
chutes deploy my_chute:chute --tag v2.0.0 --canary 10

# Monitor metrics
chutes metrics get my_chute --version v2.0.0

# Increase traffic if successful
chutes deployments promote my_chute --traffic 50

# Complete rollout
chutes deployments promote my_chute --traffic 100
```

### A/B Testing

```bash
# Deploy variant B
chutes deploy my_chute:chute --tag variant-b --variant b

# Configure traffic split
chutes deployments split my_chute --a 50 --b 50

# Monitor performance
chutes analytics compare my_chute --variants a,b
```

## Scaling and Performance

### Auto-scaling Configuration

```python
# In your chute definition
chute = Chute(
    username="myuser",
    name="my-chute",
    # Auto-scaling settings
    concurrency=4,  # Requests per instance
    min_instances=1,
    max_instances=10,
    scale_up_threshold=0.8,  # CPU utilization
    scale_down_threshold=0.2
)
```

### Manual Scaling

```bash
# Scale up to handle more traffic
chutes deployments scale my_chute --instances 5

# Scale down to reduce costs
chutes deployments scale my_chute --instances 2

# Auto-scale based on metrics
chutes deployments autoscale my_chute \
  --min 1 --max 10 \
  --cpu-target 70 \
  --memory-target 80
```

### Performance Monitoring

```bash
# Real-time metrics
chutes metrics live my_chute

# Historical performance
chutes metrics history my_chute --days 7

# Custom alerts
chutes alerts create my_chute \
  --metric response_time \
  --threshold 1000ms \
  --email `alerts@mycompany.com`
```

## Deployment Management

### Checking Deployment Status

```bash
# Quick status check
chutes deployments status my_chute

# Detailed information
chutes deployments get my_chute --detailed

# Watch real-time updates
chutes deployments logs my_chute --follow
```

### Deployment History

```bash
# List all deployments
chutes deployments list

# Show deployment history for specific chute
chutes deployments history my_chute

# Get details of specific deployment
chutes deployments get my_chute --version v1.2.0
```

### Rollback Deployments

```bash
# Rollback to previous version
chutes deployments rollback my_chute

# Rollback to specific version
chutes deployments rollback my_chute --to v1.1.0

# Emergency rollback (immediate)
chutes deployments rollback my_chute --emergency
```

## Troubleshooting Deployments

### Common Deployment Issues

**Deployment fails with "Image not found"?**

```bash
# Check if image was built
chutes images list | grep my_chute

# Build if missing
chutes build my_chute:chute --wait

# Verify image tag
chutes images get my_chute:latest
```

**Deployment stuck in "Pending" state?**

```bash
# Check resource availability
chutes resources availability

# View deployment events
chutes deployments events my_chute

# Check node requirements
chutes deployments describe my_chute
```

**Health checks failing?**

```bash
# Check chute logs
chutes deployments logs my_chute

# Test health endpoint locally
chutes run my_chute:chute --local
curl localhost:8000/health

# Verify health check configuration
chutes deployments health my_chute
```

**High latency or timeouts?**

```bash
# Check instance count
chutes deployments status my_chute

# Monitor performance
chutes metrics get my_chute --metric response_time

# Scale up if needed
chutes deployments scale my_chute --instances 3
```

### Debug Commands

```bash
# Detailed deployment logs
chutes deployments logs my_chute --debug --lines 100

# Resource utilization
chutes deployments resources my_chute

# Network connectivity test
chutes deployments ping my_chute

# Container inspection
chutes deployments exec my_chute -- ps aux
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Deploy to Chutes
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Chutes
        run: pip install chutes

      - name: Configure Chutes
        env:
          CHUTES_API_KEY: ${{ secrets.CHUTES_API_KEY }}
        run: |
          mkdir -p ~/.chutes
          echo "[auth]" > ~/.chutes/config.ini
          echo "api_key = $CHUTES_API_KEY" >> ~/.chutes/config.ini

      - name: Deploy
        run: |
          VERSION=${GITHUB_REF#refs/tags/}
          chutes deploy my_app:chute --tag $VERSION --wait

          # Health check
          sleep 30
          curl -f https://myuser-my-app.chutes.ai/health
```

### GitLab CI

```yaml
deploy:
  stage: deploy
  script:
    - pip install chutes
    - echo "[auth]" > ~/.chutes/config.ini
    - echo "api_key = $CHUTES_API_KEY" >> ~/.chutes/config.ini
    - chutes deploy my_app:chute --tag $CI_COMMIT_TAG --wait
  only:
    - tags
  environment:
    name: production
    url: https://myuser-my-app.chutes.ai
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any

    environment {
        CHUTES_API_KEY = credentials('chutes-api-key')
    }

    stages {
        stage('Deploy') {
            steps {
                sh '''
                    pip install chutes
                    mkdir -p ~/.chutes
                    echo "[auth]" > ~/.chutes/config.ini
                    echo "api_key = $CHUTES_API_KEY" >> ~/.chutes/config.ini
                    chutes deploy my_app:chute --tag $BUILD_NUMBER --wait
                '''
            }
        }

        stage('Health Check') {
            steps {
                sh '''
                    sleep 30
                    curl -f https://myuser-my-app.chutes.ai/health
                '''
            }
        }
    }
}
```

## Production Deployment Checklist

### Pre-Deployment

```bash
# ✅ Run tests
python -m pytest tests/

# ✅ Build and test image
chutes build my_chute:chute --wait
chutes run my_chute:chute --local

# ✅ Check resource requirements
chutes resources estimate my_chute:chute

# ✅ Validate configuration
chutes config validate my_chute:chute
```

### Deployment

```bash
# ✅ Deploy with specific version
VERSION=$(git describe --tags --abbrev=0)
chutes deploy my_chute:chute --tag $VERSION --wait

# ✅ Verify health
curl -f https://myuser-my-chute.chutes.ai/health

# ✅ Monitor for 5 minutes
chutes deployments logs my_chute --follow --timeout 5m
```

### Post-Deployment

```bash
# ✅ Run smoke tests
./scripts/smoke-tests.sh

# ✅ Check metrics
chutes metrics get my_chute --since 10m

# ✅ Set up monitoring alerts
chutes alerts enable my_chute

# ✅ Update documentation
git tag -a $VERSION -m "Production deployment $VERSION"
```

## Best Practices

### 1. **Version Management**

```bash
# Always tag deployments
git tag v1.2.0
chutes deploy my_chute:chute --tag v1.2.0

# Use semantic versioning
v1.0.0  # Major release
v1.1.0  # Minor update
v1.1.1  # Patch fix
```

### 2. **Environment Separation**

```bash
# Different chutes for different environments
chutes deploy my_chute_dev:chute      # Development
chutes deploy my_chute_staging:chute  # Staging
chutes deploy my_chute:chute          # Production
```

### 3. **Health Checks**

```python
# Always implement health checks
@chute.cord(public_api_path="/health", method="GET")
async def health_check(self) -> dict:
    return {
        "status": "healthy",
        "version": "1.2.0",
        "timestamp": time.time()
    }
```

### 4. **Resource Optimization**

```python
# Right-size your resources
NodeSelector(
    gpu_count=1,           # Start small
    min_vram_gb_per_gpu=16, # Match your model
    include=["rtx4090"]    # Cost-effective options
)
```

### 5. **Monitoring Setup**

```bash
# Set up comprehensive monitoring
chutes alerts create my_chute --metric error_rate --threshold 5%
chutes alerts create my_chute --metric response_time --threshold 2s
chutes alerts create my_chute --metric availability --threshold 99%
```

## Next Steps

- **[Managing Resources](/docs/cli/manage)** - Monitor and manage deployments
- **[Building Images](/docs/cli/build)** - Optimize your build process
- **[Account Management](/docs/cli/account)** - API keys and configuration
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
