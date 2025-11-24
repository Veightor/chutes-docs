# Managing Resources

This section covers CLI commands for managing your deployed chutes, monitoring performance, viewing logs, and controlling resources.

## Chute Management

### `chutes chutes list`

List all your deployed chutes.

```bash
chutes chutes list [OPTIONS]
```

**Options:**

- `--format TEXT`: Output format (table, json, yaml)
- `--filter TEXT`: Filter by name, status, or tag
- `--limit INTEGER`: Maximum number of results
- `--sort TEXT`: Sort by field (name, created, status)

**Examples:**

```bash
# List all chutes
chutes chutes list

# List with JSON output
chutes chutes list --format json

# Filter by status
chutes chutes list --filter status=running

# Filter by name pattern
chutes chutes list --filter name=*-prod
```

**Output:**

```
┌─────────────────┬─────────────────────┬────────────┬─────────────────────┐
│ Name            │ URL                 │ Status     │ Created             │
├─────────────────┼─────────────────────┼────────────┼─────────────────────┤
│ sentiment-api   │ myuser-sentiment... │ Running    │ 2024-01-15 10:30:00 │
│ image-gen       │ myuser-image-gen... │ Running    │ 2024-01-20 14:45:00 │
│ text-embeddings │ myuser-text-emb...  │ Stopped    │ 2024-01-25 09:15:00 │
└─────────────────┴─────────────────────┴────────────┴─────────────────────┘
```

### `chutes chutes get`

Get detailed information about a specific chute.

```bash
chutes chutes get <chute_name> [OPTIONS]
```

**Arguments:**

- `chute_name`: Name of the chute

**Options:**

- `--format TEXT`: Output format (table, json, yaml)
- `--show-config`: Include configuration details
- `--show-metrics`: Include performance metrics

**Examples:**

```bash
# Basic chute information
chutes chutes get my-chute

# Detailed configuration
chutes chutes get my-chute --show-config

# Include performance metrics
chutes chutes get my-chute --show-metrics
```

**Output:**

```
Chute: my-chute
├── Status: Running
├── URL: https://myuser-my-chute.chutes.ai
├── Instances: 2/2 ready
├── Image: myuser/my-chute:v1.2.0
├── GPU: 1x RTX4090 (24GB)
├── CPU: 4 cores
├── Memory: 16GB
├── Created: 2024-01-15 10:30:00
├── Last Deploy: 2024-01-20 14:45:00
└── Health: Healthy (last check: 2 minutes ago)
```

### `chutes chutes delete`

Delete a chute and all its resources.

```bash
chutes chutes delete <chute_name> [OPTIONS]
```

**Arguments:**

- `chute_name`: Name of the chute to delete

**Options:**

- `--yes`: Skip confirmation prompt
- `--keep-image`: Keep the Docker image
- `--force`: Force deletion even if chute is running

**Examples:**

```bash
# Delete with confirmation
chutes chutes delete old-chute

# Force delete without confirmation
chutes chutes delete old-chute --yes --force

# Delete but keep the image
chutes chutes delete old-chute --keep-image
```

**⚠️ Warning:** Deletion is permanent and cannot be undone!

## Logs and Monitoring

### `chutes logs`

View logs from your chute instances.

```bash
chutes logs <chute_name> [OPTIONS]
```

**Options:**

- `--follow`: Stream logs in real-time
- `--lines INTEGER`: Number of lines to show (default: 50)
- `--since TEXT`: Show logs since timestamp
- `--level TEXT`: Filter by log level (debug, info, warning, error)
- `--instance TEXT`: Show logs from specific instance
- `--download`: Download logs to file

**Examples:**

```bash
# View recent logs
chutes logs my-chute

# Follow logs in real-time
chutes logs my-chute --follow

# Show last 100 lines
chutes logs my-chute --lines 100

# Show logs from last hour
chutes logs my-chute --since 1h

# Filter error logs only
chutes logs my-chute --level error

# Download logs to file
chutes logs my-chute --download --since 24h
```

### `chutes metrics`

View performance metrics for your chutes.

```bash
chutes metrics <chute_name> [OPTIONS]
```

**Options:**

- `--metric TEXT`: Specific metric (cpu, memory, gpu, requests)
- `--timeframe TEXT`: Time range (1h, 24h, 7d, 30d)
- `--format TEXT`: Output format (table, json, graph)
- `--export TEXT`: Export to file

**Examples:**

```bash
# Overview of all metrics
chutes metrics my-chute

# CPU usage over last 24 hours
chutes metrics my-chute --metric cpu --timeframe 24h

# GPU utilization with graph
chutes metrics my-chute --metric gpu --format graph

# Export metrics to CSV
chutes metrics my-chute --export metrics.csv --timeframe 7d
```

**Sample Output:**

```
Metrics for my-chute (Last 24 hours)
┌──────────────┬─────────┬─────────┬─────────┬─────────┐
│ Metric       │ Current │ Average │ Peak    │ Min     │
├──────────────┼─────────┼─────────┼─────────┼─────────┤
│ CPU Usage    │ 45%     │ 38%     │ 82%     │ 12%     │
│ Memory Usage │ 12GB    │ 10GB    │ 15GB    │ 8GB     │
│ GPU Usage    │ 78%     │ 65%     │ 95%     │ 23%     │
│ Requests/sec │ 12.5    │ 8.3     │ 45.2    │ 0.1     │
│ Response Time│ 245ms   │ 198ms   │ 1.2s    │ 89ms    │
└──────────────┴─────────┴─────────┴─────────┴─────────┘
```

### `chutes status`

Quick status check for all or specific chutes.

```bash
chutes status [chute_name] [OPTIONS]
```

**Options:**

- `--watch`: Continuously monitor status
- `--refresh INTEGER`: Refresh interval in seconds
- `--alerts`: Show any active alerts

**Examples:**

```bash
# Status of all chutes
chutes status

# Status of specific chute
chutes status my-chute

# Watch status with auto-refresh
chutes status --watch --refresh 5

# Show status with alerts
chutes status --alerts
```

## Scaling Operations

### `chutes scale`

Scale your chute instances up or down.

```bash
chutes scale <chute_name> <instances> [OPTIONS]
```

**Arguments:**

- `chute_name`: Name of the chute
- `instances`: Target number of instances

**Options:**

- `--wait`: Wait for scaling to complete
- `--timeout INTEGER`: Timeout in seconds
- `--strategy TEXT`: Scaling strategy (immediate, rolling)

**Examples:**

```bash
# Scale up to 3 instances
chutes scale my-chute 3 --wait

# Scale down to 1 instance
chutes scale my-chute 1

# Rolling scale with timeout
chutes scale my-chute 5 --strategy rolling --timeout 300
```

### `chutes autoscale`

Configure automatic scaling for your chute.

```bash
chutes autoscale <chute_name> [OPTIONS]
```

**Options:**

- `--min INTEGER`: Minimum instances
- `--max INTEGER`: Maximum instances
- `--cpu-target INTEGER`: Target CPU utilization (%)
- `--memory-target INTEGER`: Target memory utilization (%)
- `--requests-per-second INTEGER`: Target requests per second
- `--enable/--disable`: Enable or disable autoscaling

**Examples:**

```bash
# Enable autoscaling
chutes autoscale my-chute --min 1 --max 10 --cpu-target 70

# Configure based on requests
chutes autoscale my-chute --min 2 --max 8 --requests-per-second 50

# Disable autoscaling
chutes autoscale my-chute --disable
```

## Resource Management

### `chutes resources`

View and manage resource usage.

```bash
chutes resources [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

- `usage`: Show current resource usage
- `limits`: Show account limits
- `costs`: Show cost breakdown
- `forecast`: Predict future costs

**Examples:**

```bash
# Current resource usage
chutes resources usage

# Account limits
chutes resources limits

# Cost breakdown by chute
chutes resources costs --breakdown

# 30-day cost forecast
chutes resources forecast --days 30
```

### `chutes images`

Manage your Docker images.

```bash
chutes images [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

- `list`: List all images
- `get`: Get image details
- `delete`: Delete an image
- `prune`: Remove unused images

**Examples:**

```bash
# List all images
chutes images list

# Get image details
chutes images get myuser/my-chute:v1.2.0

# Delete old image
chutes images delete myuser/my-chute:v1.0.0

# Clean up unused images
chutes images prune --older-than 30d
```

## Environment Management

### `chutes env`

Manage environment variables for your chutes.

```bash
chutes env <chute_name> [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

- `list`: List environment variables
- `set`: Set environment variable
- `unset`: Remove environment variable
- `import`: Import from file

**Examples:**

```bash
# List current environment variables
chutes env my-chute list

# Set environment variable
chutes env my-chute set DEBUG=true

# Import from file
chutes env my-chute import --file production.env

# Remove environment variable
chutes env my-chute unset DEBUG
```

## Health and Diagnostics

### `chutes health`

Check health status of your chutes.

```bash
chutes health <chute_name> [OPTIONS]
```

**Options:**

- `--detailed`: Show detailed health information
- `--history`: Show health check history
- `--alerts`: Show health-related alerts

**Examples:**

```bash
# Basic health check
chutes health my-chute

# Detailed health information
chutes health my-chute --detailed

# Health check history
chutes health my-chute --history --days 7
```

### `chutes debug`

Debug issues with your chutes.

```bash
chutes debug <chute_name> [OPTIONS]
```

**Options:**

- `--component TEXT`: Focus on specific component
- `--export`: Export debug information
- `--verbose`: Verbose output

**Examples:**

```bash
# General debug information
chutes debug my-chute

# Debug specific component
chutes debug my-chute --component networking

# Export debug bundle
chutes debug my-chute --export debug-bundle.zip
```

## Backup and Recovery

### `chutes backup`

Create backups of your chute configurations.

```bash
chutes backup <chute_name> [OPTIONS]
```

**Options:**

- `--include-data`: Include persistent data
- `--output TEXT`: Output file path
- `--compress`: Compress backup

**Examples:**

```bash
# Backup configuration
chutes backup my-chute --output my-chute-backup.json

# Full backup with data
chutes backup my-chute --include-data --compress
```

### `chutes restore`

Restore chute from backup.

```bash
chutes restore <backup_file> [OPTIONS]
```

**Options:**

- `--name TEXT`: New chute name
- `--force`: Overwrite existing chute

**Examples:**

```bash
# Restore from backup
chutes restore my-chute-backup.json

# Restore with new name
chutes restore my-chute-backup.json --name my-chute-restored
```

## Automation and Scripting

### Bash Scripting

```bash
#!/bin/bash

# Health check script
check_chute_health() {
    local chute_name=$1

    echo "Checking health of $chute_name..."

    # Get chute status
    status=$(chutes chutes get $chute_name --format json | jq -r '.status')

    if [ "$status" != "Running" ]; then
        echo "ERROR: Chute $chute_name is $status"
        return 1
    fi

    # Check health endpoint
    health=$(chutes health $chute_name --format json | jq -r '.healthy')

    if [ "$health" != "true" ]; then
        echo "ERROR: Chute $chute_name health check failed"
        return 1
    fi

    echo "SUCCESS: Chute $chute_name is healthy"
    return 0
}

# Check all chutes
chutes chutes list --format json | jq -r '.[].name' | while read chute; do
    check_chute_health $chute
done
```

### Python Scripting

```python
#!/usr/bin/env python3
import subprocess
import json
import sys

def run_chutes_command(command):
    """Run chutes CLI command and return JSON output."""
    try:
        result = subprocess.run(
            f"chutes {command}".split(),
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return None

def monitor_chutes():
    """Monitor all chutes and alert on issues."""
    chutes = run_chutes_command("chutes list --format json")

    if not chutes:
        print("Failed to get chute list")
        return

    for chute in chutes:
        name = chute['name']
        status = chute['status']

        if status != 'Running':
            print(f"ALERT: {name} is {status}")
            continue

        # Check metrics
        metrics = run_chutes_command(f"metrics {name} --format json")
        if metrics:
            cpu = metrics.get('cpu_usage', 0)
            memory = metrics.get('memory_usage', 0)

            if cpu > 90:
                print(f"ALERT: {name} high CPU usage: {cpu}%")
            if memory > 90:
                print(f"ALERT: {name} high memory usage: {memory}%")

if __name__ == "__main__":
    monitor_chutes()
```

## Performance Optimization

### Resource Right-Sizing

```bash
# Analyze resource usage
chutes metrics my-chute --timeframe 7d --export analysis.csv

# Find optimal instance count
python analyze_usage.py analysis.csv

# Apply optimizations
chutes scale my-chute 3
chutes autoscale my-chute --min 2 --max 6 --cpu-target 65
```

### Cost Optimization

```bash
# Review costs
chutes resources costs --breakdown --timeframe 30d

# Identify expensive chutes
chutes resources costs --sort-by cost --limit 10

# Optimize based on usage
chutes resources optimize --suggestions
```

## Troubleshooting

### Common Issues

**Chute not responding?**

```bash
# Check chute status
chutes chutes get my-chute

# View recent logs
chutes logs my-chute --level error --lines 50

# Check health
chutes health my-chute --detailed

# Restart if needed
chutes restart my-chute
```

**High resource usage?**

```bash
# Check metrics
chutes metrics my-chute --metric cpu,memory,gpu

# View top consumers
chutes resources usage --top 5

# Scale if needed
chutes scale my-chute 3
```

**Deployment issues?**

```bash
# Check deployment status
chutes deployments status my-chute

# View deployment logs
chutes logs my-chute --since deploy

# Debug deployment
chutes debug my-chute --component deployment
```

## Best Practices

### 1. **Regular Monitoring**

```bash
# Daily health checks
chutes health --all

# Weekly resource review
chutes resources usage --timeframe 7d

# Monthly cost analysis
chutes resources costs --breakdown --timeframe 30d
```

### 2. **Log Management**

```bash
# Regular log cleanup
chutes logs my-chute --download --since 7d
chutes logs my-chute --prune --older-than 30d

# Set up log forwarding
chutes logs my-chute --forward --endpoint https://logs.mycompany.com
```

### 3. **Backup Strategy**

```bash
# Weekly configuration backups
chutes backup my-chute --output "backup-$(date +%Y%m%d).json"

# Automated backup script
0 2 * * 0 /usr/local/bin/backup-chutes.sh
```

### 4. **Performance Tuning**

```bash
# Monitor and adjust autoscaling
chutes autoscale my-chute --cpu-target 70 --memory-target 80

# Regular performance reviews
chutes metrics my-chute --export monthly-$(date +%Y%m).csv
```

## Next Steps

- **[Building Images](/docs/cli/build)** - Optimize your images
- **[Deploying Chutes](/docs/cli/deploy)** - Advanced deployment strategies
- **[Account Management](/docs/cli/account)** - API keys and billing
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
