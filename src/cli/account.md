# Account Management

This section covers CLI commands for managing your Chutes account, registration, authentication, and API keys.

## Account Registration

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

**Examples:**

```bash
# Basic registration with interactive prompts
chutes register

# Register with specific username
chutes register --username myusername

# Register with specific wallet
chutes register --wallet my_wallet --hotkey my_hotkey
```

**Registration Process:**

1. **Choose Username**: Select a unique username for your account
2. **Wallet Configuration**: Set up Bittensor wallet for payments
3. **Verification**: Complete email verification if required
4. **Initial Setup**: Configure basic account settings

**What Happens During Registration:**

- Creates your Chutes account
- Generates initial configuration file
- Sets up billing and payment methods
- Provides developer credits for getting started

## Account Linking

### `chutes link`

Link a validator or subnet owner hotkey to your account for free developer access.

```bash
chutes link [OPTIONS]
```

**Options:**

- `--config-path TEXT`: Custom config path
- `--wallet TEXT`: Wallet name to link
- `--hotkey TEXT`: Hotkey to link
- `--force`: Force re-linking if already linked

**Examples:**

```bash
# Link with interactive prompts
chutes link

# Link specific hotkey
chutes link --wallet validator_wallet --hotkey validator_hotkey

# Force re-link existing connection
chutes link --force
```

**Benefits of Linking:**

- **Free Developer Credits**: Get additional credits for development
- **Priority Access**: Priority support and early feature access
- **Enhanced Limits**: Higher resource quotas and limits
- **Validator Benefits**: Special perks for active validators

## API Key Management

API keys provide programmatic access to your Chutes account and are essential for CI/CD and automation.

### `chutes keys list`

List all API keys for your account.

```bash
chutes keys list [OPTIONS]
```

**Options:**

- `--config-path TEXT`: Custom config path
- `--format TEXT`: Output format (table, json, yaml)

**Example:**

```bash
chutes keys list
```

**Output:**

```
┌──────────┬─────────────────────┬─────────┬─────────────────────┐
│ Name     │ ID                  │ Admin   │ Created             │
├──────────┼─────────────────────┼─────────┼─────────────────────┤
│ admin    │ key_123abc...       │ Yes     │ 2024-01-15 10:30:00 │
│ ci-cd    │ key_456def...       │ No      │ 2024-01-20 14:45:00 │
│ dev      │ key_789ghi...       │ No      │ 2024-01-25 09:15:00 │
└──────────┴─────────────────────┴─────────┴─────────────────────┘
```

### `chutes keys create`

Create a new API key.

```bash
chutes keys create [OPTIONS]
```

**Options:**

- `--name TEXT`: Name for the API key (required)
- `--admin`: Create admin key with full permissions
- `--config-path TEXT`: Custom config path
- `--expires TEXT`: Expiration date (YYYY-MM-DD)

**Examples:**

```bash
# Create basic API key
chutes keys create --name dev-key

# Create admin key with full permissions
chutes keys create --name admin --admin

# Create key with expiration
chutes keys create --name temp-key --expires 2024-12-31
```

**Key Types:**

- **Standard Keys**: Can deploy and manage your own chutes
- **Admin Keys**: Full account access including billing and user management
- **Read-Only Keys**: View-only access to account and resources

**Security Best Practices:**

```bash
# Create separate keys for different environments
chutes keys create --name production-deploy
chutes keys create --name staging-deploy
chutes keys create --name development

# Create temporary keys for contractors
chutes keys create --name contractor-temp --expires 2024-06-30

# Use read-only keys for monitoring
chutes keys create --name monitoring-readonly
```

### `chutes keys delete`

Delete an API key.

```bash
chutes keys delete <name_or_id> [OPTIONS]
```

**Arguments:**

- `name_or_id`: Name or ID of the key to delete

**Options:**

- `--config-path TEXT`: Custom config path
- `--yes`: Skip confirmation prompt

**Examples:**

```bash
# Delete by name (with confirmation)
chutes keys delete old-key

# Delete by ID
chutes keys delete key_123abc456def

# Delete without confirmation
chutes keys delete temp-key --yes
```

**Safety Notes:**

- Deleted keys cannot be recovered
- Active deployments using the key will lose access
- Always rotate keys before deletion in production

## Configuration Management

### Config File Structure

The Chutes configuration file (`~/.chutes/config.ini`) stores your account settings:

```ini
[account]
username = myusername
user_id = user_123abc456def

[auth]
api_key = key_your_api_key_here

[wallet]
wallet_name = my_wallet
hotkey_name = my_hotkey

[settings]
default_region = us-east
debug = false
```

### Environment Variables

Override config settings with environment variables:

```bash
# API Configuration
export CHUTES_API_KEY=your_api_key_here
export CHUTES_API_URL=https://api.chutes.ai

# Account Settings
export CHUTES_USERNAME=myusername
export CHUTES_DEFAULT_REGION=us-west

# Development Settings
export CHUTES_DEBUG=true
export CHUTES_DEV_MODE=true
```

### Multiple Configurations

Manage multiple accounts or environments:

```bash
# Create environment-specific configs
mkdir -p ~/.chutes/environments

# Production config
chutes register --config-path ~/.chutes/environments/prod.ini

# Staging config
chutes register --config-path ~/.chutes/environments/staging.ini

# Use specific config
chutes build my_app:chute --config-path ~/.chutes/environments/prod.ini
```

## Account Information

### View Account Details

```bash
# Show current account info
chutes account info

# Show account usage and billing
chutes account usage

# Show account limits
chutes account limits
```

### Account Settings

```bash
# Update account settings
chutes account update --email `new@example.com`

# Change password
chutes account password

# Update billing information
chutes account billing
```

## Troubleshooting

### Common Issues

**Registration fails?**

```bash
# Check network connectivity
curl -I https://api.chutes.ai

# Try with different username
chutes register --username alternative_username

# Check wallet configuration
chutes wallet verify
```

**API key not working?**

```bash
# Verify key is active
chutes keys list

# Test key permissions
chutes auth test

# Check key hasn't expired
chutes keys get my-key
```

**Configuration issues?**

```bash
# Validate configuration
chutes config validate

# Reset configuration
chutes config reset

# Show current config
chutes config show
```

### Getting Help

- **Account Issues**: `support@chutes.ai`
- **Billing Questions**: `support@chutes.ai`
- **Technical Support**: [Discord Community](https://discord.gg/wHrXwWkCRz)
- **Documentation**: [Chutes Docs](https://chutes.ai/docs)

## Security Best Practices

### API Key Security

```bash
# Rotate keys regularly
chutes keys create --name new-prod-key
# Update deployments to use new key
chutes keys delete old-prod-key

# Use least privilege
chutes keys create --name readonly-monitoring  # No admin flag

# Set expiration dates
chutes keys create --name contractor --expires 2024-06-30
```

### Account Security

- **Enable 2FA**: Add two-factor authentication to your account
- **Regular Audits**: Review API keys and access regularly
- **Secure Storage**: Never commit API keys to version control
- **Environment Separation**: Use different keys for dev/staging/prod

### CI/CD Security

```yaml
# GitHub Actions example
env:
  CHUTES_API_KEY: ${{ secrets.CHUTES_API_KEY }} # Store in secrets
  CHUTES_CONFIG_PATH: /tmp/chutes-config.ini

steps:
  - name: Configure Chutes
    run: |
      mkdir -p ~/.chutes
      echo "[auth]" > ~/.chutes/config.ini
      echo "api_key = $CHUTES_API_KEY" >> ~/.chutes/config.ini
```

## Next Steps

- **[Building Images](/docs/cli/build)** - Learn to build Docker images
- **[Deploying Chutes](/docs/cli/deploy)** - Deploy your applications
- **[Managing Resources](/docs/cli/manage)** - Manage your deployments
- **[CLI Overview](/docs/cli/overview)** - Return to command overview
