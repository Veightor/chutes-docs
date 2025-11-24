# Installation & Setup

This guide will walk you through installing the Chutes SDK and setting up your development environment.

## Prerequisites

Before installing Chutes, ensure you have:

- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **pip** package manager
- A **Bittensor wallet** (required for authentication)

## Installing the Chutes SDK

### Option 1: Install from PyPI (Recommended)

```bash
pip install chutes
```

### Option 2: Install from Source

```bash
git clone https://github.com/rayonlabs/chutes.git
cd chutes
pip install -e .
```

### Verify Installation

Check that Chutes was installed correctly:

```bash
chutes --help
```

You should see the Chutes CLI help menu.

## Setting Up Authentication

Chutes uses **Bittensor** for secure authentication. You'll need a Bittensor wallet with a hotkey.

### Creating a Bittensor Wallet

If you don't already have a Bittensor wallet:

#### Option 1: Automatic Setup (Recommended)

Visit [chutes.ai](https://chutes.ai) and create an account. The platform will automatically create and manage your wallet for you.

#### Option 2: Manual Setup

If you prefer to manage your own wallet:

1. Install Bittensor (older version required):

   ```bash
   pip install 'bittensor<8'
   ```

2. Create a coldkey and hotkey:

   ```bash
   # Create a coldkey (your main wallet)
   btcli wallet new_coldkey --n_words 24 --wallet.name my-chutes-wallet

   # Create a hotkey (for signing transactions)
   btcli wallet new_hotkey --wallet.name my-chutes-wallet --n_words 24 --wallet.hotkey my-hotkey
   ```

### Registering with Chutes

Once you have a Bittensor wallet, register with the Chutes platform:

```bash
chutes register
```

Follow the interactive prompts to:

1. Enter your desired username
2. Select your Bittensor wallet
3. Choose your hotkey
4. Complete the registration process

After successful registration, you'll find your configuration at `~/.chutes/config.ini`.

## Configuration

Your Chutes configuration is stored in `~/.chutes/config.ini`:

```ini
[auth]
user_id = your-user-id
username = your-username
hotkey_seed = your-hotkey-seed
hotkey_name = your-hotkey-name
hotkey_ss58address = your-hotkey-address

[api]
base_url = https://api.chutes.ai
```

### Environment Variables

You can override configuration with environment variables:

```bash
export CHUTES_CONFIG_PATH=/custom/path/to/config.ini
export CHUTES_API_URL=https://api.chutes.ai
export CHUTES_DEV_URL=http://localhost:8000  # For local development
```

## Creating API Keys

For programmatic access, create API keys:

### Full Admin Access

```bash
chutes keys create --name admin-key --admin
```

### Limited Access

```bash
# Access to specific chutes (requires action parameter)
chutes keys create --name my-app-key --chute-ids <chute-id> --action read

# Access to images only (requires action parameter)
chutes keys create --name image-key --images --action write
```

### Using API Keys

Use your API keys in HTTP requests:

```bash
curl -H "Authorization: Bearer cpk_your_api_key" \
     https://api.chutes.ai/chutes/
```

Or in Python:

```python
import aiohttp

headers = {"Authorization": "Basic cpk_your_api_key"}
async with aiohttp.ClientSession() as session:
    async with session.get("https://api.chutes.ai/chutes/", headers=headers) as resp:
        data = await resp.json()
```

## Developer Deposit (Optional)

To create and deploy your own chutes and images, you'll need to make a developer deposit. This is a refundable security deposit to prevent spam.

### Check Required Deposit

```bash
curl -s https://api.chutes.ai/developer_deposit | jq .
```

### Get Your Deposit Address

```bash
curl -s https://api.chutes.ai/users/me \
  -H 'authorization: Basic cpk_your_api_key'
```

### Return Your Deposit

After 7 days, you can request your deposit back:

```bash
curl -XPOST https://api.chutes.ai/return_developer_deposit \
  -H 'content-type: application/json' \
  -H 'authorization: Basic cpk_your_api_key' \
  -d '{"address": "your-deposit-address"}'
```

## IDE Setup

### VS Code

For the best development experience with VS Code:

1. Install the **Python extension**
2. Set up your Python interpreter to use the environment where you installed Chutes
3. Add this to your `.vscode/settings.json`:

```json
{
	"python.linting.enabled": true,
	"python.linting.pylintEnabled": true,
	"python.formatting.provider": "black",
	"python.analysis.typeCheckingMode": "basic"
}
```

### PyCharm

For PyCharm users:

1. Configure your Python interpreter
2. Add Chutes to your project dependencies
3. Enable type checking for better IntelliSense

## Troubleshooting

### Common Issues

**"Command not found: chutes"**

- Make sure your Python `Scripts` directory is in your `PATH`
- Try `python -m chutes` instead

**"Invalid hotkey" during registration**

- Ensure your Bittensor wallet is properly created
- Check that you're using the correct wallet and hotkey names

**"Permission denied" errors**

- You might need to use `sudo` on some systems
- Consider using a virtual environment

**"API connection failed"**

- Check your internet connection
- Verify the API URL in your config
- Ensure you have the latest version of Chutes

### Getting Help

If you encounter issues:

1. Check the [FAQ](/docs/help/faq)
2. Search existing [GitHub issues](https://github.com/rayonlabs/chutes/issues)
3. Join our [Discord community](https://discord.gg/wHrXwWkCRz)
4. Email `support@chutes.ai`

## Next Steps

Now that you have Chutes installed and configured:

1. **[Quick Start Guide](/docs/getting-started/quickstart)** - Deploy your first chute in minutes
2. **[Your First Chute](/docs/getting-started/first-chute)** - Build a complete application from scratch
3. **[Core Concepts](/docs/core-concepts/chutes)** - Understand the fundamentals

---

Ready to build something amazing? Let's move on to the [Quick Start Guide](/docs/getting-started/quickstart)!
