# Authentication & Account Setup

This guide covers setting up authentication for the Chutes platform using Bittensor wallets and managing API keys.

## Overview

Chutes uses **Bittensor** for secure, decentralized authentication. This provides:

- 🔐 **Cryptographic Security**: Wallet-based authentication
- 🌐 **Decentralized Identity**: No central password database
- 🔑 **API Key Management**: Granular access control
- 💰 **Integrated Billing**: Seamless payment integration

## Bittensor Wallet Setup

### Option 1: Automatic Setup (Recommended)

The easiest way to get started:

1. **Visit [chutes.ai](https://chutes.ai)**
2. **Click "Create Account"**
3. **Follow the guided setup**

The platform will automatically:

- Create your Bittensor wallet
- Generate secure keys
- Set up your account
- Provide you with wallet credentials

### Option 2: Manual Wallet Creation

If you prefer to manage your own wallet:

#### Install Bittensor

```bash
# Install older version (required for wallet creation)
pip install 'bittensor<8'
```

> **Note**: We use an older Bittensor version because newer versions require Rust compilation, which can be complex to set up.

#### Create Wallet and Hotkey

```bash
# Create a coldkey (your main wallet)
btcli wallet new_coldkey \
  --n_words 24 \
  --wallet.name chutes-wallet

# Create a hotkey (for signing transactions)
btcli wallet new_hotkey \
  --wallet.name chutes-wallet \
  --wallet.hotkey default \
  --n_words 24
```

#### Secure Your Keys

```bash
# Your wallets are stored in:
ls ~/.bittensor/wallets/

# Back up your coldkey and hotkey files
# Store them securely - they cannot be recovered if lost!
```

## Account Registration

Once you have a Bittensor wallet, register with Chutes:

```bash
chutes register
```

### Interactive Registration Process

The registration wizard will ask for:

1. **Username**: Your desired Chutes username
2. **Wallet Selection**: Choose from available wallets
3. **Hotkey Selection**: Choose from available hotkeys
4. **Confirmation**: Verify your selections

### Example Registration Session

```bash
$ chutes register
Enter desired username: myawesomeai
Found wallets: ['chutes-wallet', 'other-wallet']
Select wallet (chutes-wallet): chutes-wallet
Found hotkeys: ['default', 'backup']
Select hotkey (default): default
✅ Registration successful!
```

## Configuration File

After registration, you'll find your config at `~/.chutes/config.ini`:

```ini
[auth]
user_id = usr_1234567890abcdef
username = myawesomeai
hotkey_seed = your-encrypted-hotkey-seed
hotkey_name = default
hotkey_ss58address = 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY

[api]
base_url = https://api.chutes.ai
```

### Environment Variable Overrides

You can override configuration with environment variables:

```bash
# Custom config location
export CHUTES_CONFIG_PATH=/custom/path/config.ini

# Custom API endpoint
export CHUTES_API_URL=https://api.chutes.ai

# Development mode
export CHUTES_DEV_URL=http://localhost:8000
```

## API Key Management

For programmatic access and CI/CD, create API keys:

### Creating API Keys

#### Full Administrative Access

```bash
chutes keys create --name admin-key --admin
```

#### Scoped Access Examples

```bash
# Access to specific chutes only
chutes keys create \
  --name my-app-key \
  --chute-ids 12345678-1234-5678-9abc-123456789012 \
  --action invoke

# Read-only access to images
chutes keys create \
  --name readonly-images \
  --images \
  --action read

# Multiple chute access
chutes keys create \
  --name multi-chute-key \
  --chute-ids 12345678-1234-5678-9abc-123456789012,87654321-4321-8765-cba9-210987654321 \
  --action invoke
```

#### Advanced Scoping

```bash
# JSON-based scoping for complex permissions
chutes keys create \
  --name complex-key \
  --json-input '{
    "scopes": [
      {"object_type": "chutes", "action": "invoke"},
      {"object_type": "images", "action": "read", "object_id": "specific-image-id"}
    ]
  }'
```

### Using API Keys

#### HTTP Requests

```bash
curl -H "Authorization: Bearer cpk_your_api_key_here" \
     https://api.chutes.ai/chutes/
```

#### Python SDK

```python
import aiohttp

async def call_chutes_api():
    headers = {"Authorization": "Basic cpk_your_api_key_here"}

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.chutes.ai/chutes/",
            headers=headers
        ) as response:
            return await response.json()
```

#### Environment Variables

```bash
# Set API key as environment variable
export CHUTES_API_KEY=cpk_your_api_key_here

# Use in scripts
curl -H "Authorization: Bearer $CHUTES_API_KEY" \
     https://api.chutes.ai/chutes/
```

### Managing API Keys

#### List Your Keys

```bash
chutes keys list
```

#### View Key Details

```bash
chutes keys get my-app-key
```

#### Delete Keys

```bash
chutes keys delete old-key-name
```

## Developer Deposit

To create and deploy chutes, you need a refundable developer deposit:

### Check Required Deposit

```bash
curl -s https://api.chutes.ai/developer_deposit | jq .
```

### Get Your Deposit Address

```bash
curl -s https://api.chutes.ai/users/me \
  -H "Authorization: Bearer cpk_your_api_key" | jq .deposit_address
```

### Making the Deposit

1. **Get your deposit address** from the API call above
2. **Transfer TAO** to that address using your preferred wallet
3. **Wait for confirmation** (usually 1-2 blocks)
4. **Verify deposit** status in your account

### Returning Your Deposit

After at least 7 days:

```bash
curl -X POST https://api.chutes.ai/return_developer_deposit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cpk_your_api_key" \
  -d '{"address": "5EcZsewZSTxUaX8gwyHzkKsqT3NwLP1n2faZPyjttCeaPdYe"}'
```

## Free Developer Access

### Validator/Subnet Owner Benefits

If you own a validator or subnet on Bittensor, you can get free developer access:

```bash
chutes link
```

This will:

- Link your validator hotkey to your account
- Grant free access to Chutes features
- Bypass the developer deposit requirement

### Eligibility Requirements

- Must own an active validator on Bittensor
- Or be a subnet owner
- Hotkey must be currently registered and active

## Security Best Practices

### Wallet Security

1. **Backup Your Keys**

   ```bash
   # Create secure backups
   cp -r ~/.bittensor/wallets/ /secure/backup/location/
   ```

2. **Use Separate Hotkeys**

   ```bash
   # Create dedicated hotkeys for different purposes
   btcli wallet new_hotkey --wallet.name chutes-wallet --wallet.hotkey production
   btcli wallet new_hotkey --wallet.name chutes-wallet --wallet.hotkey development
   ```

3. **Secure Storage**
   - Store coldkey offline when possible
   - Use hardware wallets for large amounts
   - Never share your seed phrases

### API Key Security

1. **Principle of Least Privilege**

   ```bash
   # Create keys with minimal required permissions
   chutes keys create --name limited-key --chute-ids specific-id --action read
   ```

2. **Regular Rotation**

   ```bash
   # Rotate keys regularly
   chutes keys delete old-key
   chutes keys create --name new-key --admin
   ```

3. **Environment Management**
   ```bash
   # Use environment variables, never hardcode keys
   export CHUTES_API_KEY=cpk_your_key_here
   # Add to .env files, not source code
   ```

## Troubleshooting

### Common Authentication Issues

#### "Invalid hotkey" Error

```bash
# Check wallet status
btcli wallet list

# Verify hotkey registration
btcli wallet overview --wallet.name your-wallet
```

#### "Config not found" Error

```bash
# Check config location
echo $CHUTES_CONFIG_PATH
ls -la ~/.chutes/

# Re-register if needed
chutes register
```

#### "API key invalid" Error

```bash
# Verify key exists
chutes keys list

# Check key permissions
chutes keys get your-key-name

# Test key
curl -H "Authorization: Bearer cpk_your_key" \
     https://api.chutes.ai/users/me
```

### Network Issues

#### API Connection Problems

```bash
# Test API connectivity
curl -v https://api.chutes.ai/ping

# Check DNS resolution
nslookup api.chutes.ai

# Try alternative endpoints
export CHUTES_API_URL=https://backup.api.chutes.ai
```

### Wallet Issues

#### Bittensor Installation Problems

```bash
# Install specific version
pip install bittensor==7.3.0

# Clear cache if needed
pip cache purge
pip install --no-cache-dir 'bittensor<8'
```

#### Permission Errors

```bash
# Fix wallet permissions
chmod 600 ~/.bittensor/wallets/*/coldkey
chmod 600 ~/.bittensor/wallets/*/hotkeys/*
```

## Next Steps

Now that authentication is set up:

1. **[Quick Start Guide](quickstart)** - Deploy your first chute
2. **[Your First Custom Chute](first-chute)** - Build from scratch
3. **[API Key Management](../cli/account)** - Advanced key management
4. **[Security Best Practices](../guides/best-practices)** - Production security

## Getting Help

- 📖 **Documentation**: [Installation Guide](installation)
- 💬 **Discord**: [Community Support](https://discord.gg/wHrXwWkCRz)
- 🐛 **Issues**: [GitHub Issues](https://github.com/rayonlabs/chutes/issues)
- 📧 **Support**: `support@chutes.ai`

---

**Authentication set up?** Great! Now head to the [Quick Start Guide](quickstart) to deploy your first chute.
