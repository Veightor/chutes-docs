# CLI Command Overview

The Chutes CLI provides a complete set of commands for managing your AI applications, from account setup to deployment and monitoring.

## Installation

The CLI is included when you install the Chutes SDK:

```bash
pip install chutes
```

Verify installation:

```bash
chutes --help
```

## Command Structure

All Chutes commands follow this pattern:

```bash
chutes <command> [subcommand] [options] [arguments]
```

## Account Management

### `chutes register`

Create a new account with the Chutes platform.

```bash
chutes register [OPTIONS]
```

**Options:**

- `--config-path TEXT`: Custom path to config file
- `--username TEXT`: Desired username
- `--wallets-path TEXT`: Path to Bittensor wallets directory
- `--wallet TEXT`: Name of the wallet to use
- `--hotkey TEXT`: Hotkey to register with

**Example:**

```bash
chutes register --username myuser
```

### `chutes link`

Link a validator or subnet owner hotkey to your account for free developer access.

```bash
chutes link [OPTIONS]
```

## Building & Deployment

### `chutes build`

Build a Docker image for your chute.

```bash
chutes build <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--config-path TEXT`: Custom config path
- `--logo TEXT`: Path to logo image
- `--local`: Build locally instead of remotely
- `--debug`: Enable debug logging
- `--include-cwd`: Include entire current directory
- `--wait`: Wait for build to complete
- `--public`: Mark image as public

**Examples:**

```bash
# Build remotely and wait for completion
chutes build my_chute:chute --wait

# Build locally for testing
chutes build my_chute:chute --local

# Build with a logo
chutes build my_chute:chute --logo ./logo.png --public
```

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

**Examples:**

```bash
# Basic deployment
chutes deploy my_chute:chute

# Deploy with logo
chutes deploy my_chute:chute --logo ./logo.png

# Deploy as public chute
chutes deploy my_chute:chute --public
```

### `chutes run`

Run a chute locally for development and testing.

```bash
chutes run <chute_ref> [OPTIONS]
```

**Arguments:**

- `chute_ref`: Chute reference in format `module:chute_name`

**Options:**

- `--host TEXT`: Host to bind to (default: 0.0.0.0)
- `--port INTEGER`: Port to listen on (default: 8000)
- `--debug`: Enable debug logging
- `--dev`: Enable development mode

**Examples:**

```bash
# Run on default port
chutes run my_chute:chute

# Run on custom port with debug
chutes run my_chute:chute --port 8080 --debug

# Development mode
chutes run my_chute:chute --dev
```

## Resource Management

### `chutes chutes`

Manage your deployed chutes.

#### `chutes chutes list`

List your chutes.

```bash
chutes chutes list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page (default: 25)
- `--page INTEGER`: Page number (default: 0)
- `--include-public`: Include public chutes

**Example:**

```bash
chutes chutes list --limit 10 --include-public
```

#### `chutes chutes get`

Get detailed information about a specific chute.

```bash
chutes chutes get <name_or_id>
```

**Example:**

```bash
chutes chutes get my-awesome-chute
```

#### `chutes chutes delete`

Delete a chute.

```bash
chutes chutes delete <name_or_id>
```

**Example:**

```bash
chutes chutes delete my-old-chute
```

### `chutes images`

Manage your Docker images.

#### `chutes images list`

List your images.

```bash
chutes images list [OPTIONS]
```

**Options:**

- `--name TEXT`: Filter by name
- `--limit INTEGER`: Number of items per page
- `--page INTEGER`: Page number
- `--include-public`: Include public images

#### `chutes images get`

Get detailed information about a specific image.

```bash
chutes images get <name_or_id>
```

#### `chutes images delete`

Delete an image.

```bash
chutes images delete <name_or_id>
```

### `chutes keys`

Manage API keys.

#### `chutes keys create`

Create a new API key.

```bash
chutes keys create [OPTIONS]
```

**Options:**

- `--name TEXT`: Name for the API key (required)
- `--admin`: Grant admin access
- `--images`: Grant access to images
- `--chutes`: Grant access to chutes
- `--image-ids TEXT`: Specific image IDs to allow
- `--chute-ids TEXT`: Specific chute IDs to allow
- `--action [read|write|delete|invoke]`: Specify action scope

**Examples:**

```bash
# Admin key
chutes keys create --name admin-key --admin

# Read-only access to specific chute
chutes keys create --name readonly-key --chute-ids 12345 --action read

# Image management key
chutes keys create --name image-key --images
```

#### `chutes keys list`

List your API keys.

```bash
chutes keys list [OPTIONS]
```

#### `chutes keys get`

Get details about a specific API key.

```bash
chutes keys get <name_or_id>
```

#### `chutes keys delete`

Delete an API key.

```bash
chutes keys delete <name_or_id>
```

## Utilities

### `chutes report`

Report an invocation for billing/tracking purposes.

```bash
chutes report [OPTIONS]
```

### `chutes refinger`

Change your fingerprint.

```bash
chutes refinger [OPTIONS]
```

## Global Options

These options work with most commands:

- `--help`: Show help message
- `--config-path TEXT`: Path to custom config file
- `--debug`: Enable debug logging

## Exit Codes

The CLI uses standard exit codes:

- `0`: Success
- `1`: General error
- `2`: Argument error
- `130`: Interrupted by user (Ctrl+C)

## Configuration

### Config File Location

Default: `~/.chutes/config.ini`

Override with:

```bash
export CHUTES_CONFIG_PATH=/path/to/config.ini
```

### Environment Variables

- `CHUTES_CONFIG_PATH`: Custom config file path
- `CHUTES_API_URL`: API base URL
- `CHUTES_DEV_URL`: Development server URL
- `CHUTES_ALLOW_MISSING`: Allow missing config

## Common Workflows

### 1. First-Time Setup

```bash
# Register account
chutes register

# Create admin API key
chutes keys create --name admin --admin
```

### 2. Develop and Deploy

```bash
# Build your image
chutes build my_app:chute --wait

# Test locally
chutes run my_app:chute --dev

# Deploy to production
chutes deploy my_app:chute
```

### 3. Manage Resources

```bash
# List your chutes
chutes chutes list

# Get detailed info
chutes chutes get my-app

# Check logs via dashboard
# (Visit https://chutes.ai)

# Clean up old resources
chutes chutes delete old-chute
chutes images delete old-image
```

## Troubleshooting

### Common Issues

**Command not found**

```bash
# Check installation
pip show chutes

# Try with Python module
python -m chutes --help
```

**Authentication errors**

```bash
# Re-register if needed
chutes register

# Check config file
cat ~/.chutes/config.ini
```

**Build failures**

```bash
# Try local build for debugging
chutes build my_app:chute --local --debug

# Check image syntax
python -c "from my_app import chute; print(chute.image)"
```

**Deployment issues**

```bash
# Verify image exists
chutes images list --name my-image

# Check chute status
chutes chutes get my-chute
```

### Debug Mode

Enable debug logging for detailed output:

```bash
chutes --debug <command>
```

Or set environment variable:

```bash
export CHUTES_DEBUG=1
chutes <command>
```

## Getting Help

### Built-in Help

```bash
# General help
chutes --help

# Command-specific help
chutes build --help
chutes deploy --help
chutes chutes list --help
```

### Support Resources

- 📖 **Documentation**: [Complete Docs](/docs)
- 💬 **Discord**: [Community Chat](https://discord.gg/wHrXwWkCRz)
- 🐛 **Issues**: [GitHub Issues](https://github.com/rayonlabs/chutes/issues)
- 📧 **Email**: `support@chutes.ai`

## Advanced Usage

### Scripting and Automation

The CLI is designed for scripting:

```bash
#!/bin/bash
set -e

echo "Building and deploying my chute..."

# Build
chutes build my_app:chute --wait || exit 1

# Deploy
chutes deploy my_app:chute || exit 1

# Verify deployment
chutes chutes get my-app

echo "Deployment successful!"
```

### JSON Output

Many commands support JSON output for programmatic use:

```bash
# Get chute info as JSON
chutes chutes get my-chute --format json

# List chutes with jq processing
chutes chutes list --format json | jq '.items[].name'
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy Chute
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install Chutes
        run: pip install chutes

      - name: Configure Chutes
        run: |
          mkdir -p ~/.chutes
          echo "${{ secrets.CHUTES_CONFIG }}" > ~/.chutes/config.ini

      - name: Deploy
        run: |
          chutes build my_app:chute --wait
          chutes deploy my_app:chute
```

---

Continue to specific command documentation:

- **[Account Management](/docs/cli/account)** - Detailed account commands
- **[Building Images](/docs/cli/build)** - Advanced build options
- **[Deploying Chutes](/docs/cli/deploy)** - Deployment strategies
- **[Managing Resources](/docs/cli/manage)** - Resource management
