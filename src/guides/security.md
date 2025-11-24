# Security Guide

This comprehensive guide covers security best practices for Chutes applications, from basic security measures to advanced threat protection.

## Overview

Security in Chutes involves multiple layers:

- **Authentication & Authorization**: Secure API access and user management
- **Data Protection**: Encrypting sensitive data and communications
- **Container Security**: Securing Docker images and runtime environments
- **Network Security**: Protecting network communications
- **Monitoring & Incident Response**: Detecting and responding to security threats

## Authentication & Authorization

### API Key Management

Secure API key handling:

```python
import os
import hashlib
import hmac
import time
from typing import Optional

class APIKeyManager:
    def __init__(self):
        self.secret_key = os.environ.get("API_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("API_SECRET_KEY environment variable is required")

    def generate_api_key(self, user_id: str) -> str:
        """Generate secure API key for user"""
        timestamp = str(int(time.time()))
        payload = f"{user_id}:{timestamp}"

        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}:{signature}"

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user_id if valid"""
        try:
            parts = api_key.split(":")
            if len(parts) != 3:
                return None

            user_id, timestamp, signature = parts
            payload = f"{user_id}:{timestamp}"

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return None

            # Check if key is expired (24 hours)
            key_age = time.time() - int(timestamp)
            if key_age > 86400:  # 24 hours
                return None

            return user_id

        except Exception:
            return None

# Use in chute
api_manager = APIKeyManager()

async def authenticate_request(headers: dict) -> Optional[str]:
    """Authenticate incoming request"""
    auth_header = headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    api_key = auth_header[7:]  # Remove "Bearer " prefix
    return api_manager.validate_api_key(api_key)

async def run_secure(inputs: dict) -> dict:
    """Secure endpoint with authentication"""
    headers = inputs.get("headers", {})
    user_id = await authenticate_request(headers)

    if not user_id:
        return {"error": "Unauthorized", "status": 401}

    # Process authenticated request
    result = await process_for_user(user_id, inputs)
    return {"result": result, "user_id": user_id}
```

### Role-Based Access Control

Implement authorization:

```python
from enum import Enum
from typing import List, Set
import json

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class Role:
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

class RBACManager:
    def __init__(self):
        # Define roles
        self.roles = {
            "user": Role("user", {Permission.READ}),
            "editor": Role("editor", {Permission.READ, Permission.WRITE}),
            "admin": Role("admin", {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN})
        }

        # User role assignments (in production, store in database)
        self.user_roles = {}

    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")

        self.user_roles[user_id] = role_name

    def check_permission(self, user_id: str, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        role_name = self.user_roles.get(user_id)
        if not role_name:
            return False

        role = self.roles.get(role_name)
        if not role:
            return False

        return required_permission in role.permissions

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract user_id from inputs
                inputs = args[0] if args else kwargs.get("inputs", {})
                user_id = inputs.get("user_id")

                if not user_id or not self.check_permission(user_id, permission):
                    return {"error": "Forbidden", "status": 403}

                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Global RBAC manager
rbac = RBACManager()

@rbac.require_permission(Permission.WRITE)
async def create_resource(inputs: dict) -> dict:
    """Endpoint that requires write permission"""
    # Create resource logic
    return {"message": "Resource created successfully"}

@rbac.require_permission(Permission.ADMIN)
async def admin_operation(inputs: dict) -> dict:
    """Admin-only endpoint"""
    # Admin operation logic
    return {"message": "Admin operation completed"}
```

## Data Protection

### Input Validation & Sanitization

Prevent injection attacks:

```python
import re
import html
from typing import Any, Dict
from pydantic import BaseModel, validator, Field

class SecureInput(BaseModel):
    text: str = Field(..., max_length=10000)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    filename: str = Field(..., regex=r'^[a-zA-Z0-9._-]+$')

    @validator('text')
    def sanitize_text(cls, v):
        """Sanitize text input"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', v)
        # HTML escape
        sanitized = html.escape(sanitized)
        return sanitized

    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename for path traversal"""
        # Prevent path traversal
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("Invalid filename")
        return v

class InputSanitizer:
    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        """Sanitize input to prevent SQL injection"""
        # Remove SQL keywords and special characters
        dangerous_patterns = [
            r'(\bUNION\b)|(\bSELECT\b)|(\bINSERT\b)|(\bUPDATE\b)|(\bDELETE\b)',
            r'(\bDROP\b)|(\bCREATE\b)|(\bALTER\b)|(\bEXEC\b)',
            r'[;\'"`]'
        ]

        sanitized = value
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file path to prevent directory traversal"""
        # Remove dangerous path components
        sanitized = re.sub(r'\.\.+', '', path)
        sanitized = re.sub(r'[/\\]+', '_', sanitized)
        return sanitized

async def run_with_validation(inputs: dict) -> dict:
    """Validate and sanitize all inputs"""
    try:
        # Validate using Pydantic model
        validated_input = SecureInput(**inputs)

        # Additional sanitization
        sanitizer = InputSanitizer()

        if "file_path" in inputs:
            inputs["file_path"] = sanitizer.sanitize_file_path(inputs["file_path"])

        # Process with sanitized inputs
        result = await process_secure_inputs(validated_input.dict())
        return {"result": result}

    except Exception as e:
        return {"error": f"Invalid input: {str(e)}", "status": 400}
```

### Data Encryption

Encrypt sensitive data:

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: str = None):
        self.password = password or os.environ.get("ENCRYPTION_PASSWORD")
        if not self.password:
            raise ValueError("Encryption password is required")

        # Generate key from password
        self.key = self._generate_key(self.password)
        self.cipher = Fernet(self.key)

    def _generate_key(self, password: str) -> bytes:
        """Generate encryption key from password"""
        # Use a fixed salt for consistent keys (in production, use random salt per data)
        salt = b'chutes_security_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000)
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return decrypted_data.decode()

    def encrypt_dict(self, data: dict, sensitive_fields: list) -> dict:
        """Encrypt sensitive fields in dictionary"""
        encrypted_data = data.copy()

        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))

        return encrypted_data

    def decrypt_dict(self, data: dict, sensitive_fields: list) -> dict:
        """Decrypt sensitive fields in dictionary"""
        decrypted_data = data.copy()

        for field in sensitive_fields:
            if field in decrypted_data:
                decrypted_data[field] = self.decrypt(decrypted_data[field])

        return decrypted_data

# Global encryption instance
encryption = DataEncryption()

async def run_with_encryption(inputs: dict) -> dict:
    """Handle sensitive data with encryption"""
    sensitive_fields = ["personal_info", "api_keys", "passwords"]

    # Encrypt sensitive inputs
    encrypted_inputs = encryption.encrypt_dict(inputs, sensitive_fields)

    # Process with encrypted data
    result = await process_encrypted_data(encrypted_inputs)

    # Decrypt result if needed
    if "sensitive_result" in result:
        result["sensitive_result"] = encryption.decrypt(result["sensitive_result"])

    return result
```

## Container Security

### Secure Docker Images

Build secure container images:

```python
from chutes.image import Image

# Security-hardened image
secure_image = (
    Image(
        username="myuser",
        name="secure-app",
        tag="hardened",
        base_image="python:3.11-slim",  # Use minimal base image
        python_version="3.11"
    )
    # Create non-root user
    .run_command("""
        groupadd -r appuser && \\
        useradd -r -g appuser -d /app -s /sbin/nologin appuser && \\
        mkdir -p /app && \\
        chown -R appuser:appuser /app
    """)

    # Install security updates
    .run_command("""
        apt-get update && \\
        apt-get upgrade -y && \\
        apt-get install -y --no-install-recommends \\
        ca-certificates && \\
        apt-get clean && \\
        rm -rf /var/lib/apt/lists/*
    """)

    # Install Python dependencies with security focus
    .pip_install([
        "cryptography==41.0.7",  # Pin specific versions
        "pydantic==2.4.2",
        "bcrypt==4.0.1"
    ])

    # Copy application code with proper ownership
    .copy_files("./app", "/app", owner="appuser:appuser")

    # Set secure permissions
    .run_command("chmod -R 755 /app")

    # Security configurations
    .set_environment_variable("PYTHONUNBUFFERED", "1")
    .set_environment_variable("PYTHONDONTWRITEBYTECODE", "1")
    .set_environment_variable("PYTHONHASHSEED", "random")

    # Switch to non-root user
    .set_user("appuser")
    .set_working_directory("/app")
)
```

### Runtime Security

Implement runtime security measures:

```python
import os
import sys
import signal
import logging
from contextlib import contextmanager

class SecurityManager:
    def __init__(self):
        self.setup_logging()
        self.setup_signal_handlers()
        self.validate_environment()

    def setup_logging(self):
        """Configure secure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/app/logs/security.log', mode='a')
            ]
        )
        self.logger = logging.getLogger('security')

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            self.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def validate_environment(self):
        """Validate security environment variables"""
        required_vars = ["API_SECRET_KEY", "ENCRYPTION_PASSWORD"]

        for var in required_vars:
            if not os.environ.get(var):
                self.logger.error(f"Required environment variable {var} is missing")
                raise ValueError(f"Missing required environment variable: {var}")

    def log_security_event(self, event_type: str, details: dict):
        """Log security events"""
        self.logger.warning(f"SECURITY EVENT: {event_type} - {details}")

    @contextmanager
    def secure_execution(self):
        """Context manager for secure code execution"""
        try:
            self.logger.info("Starting secure execution")
            yield
        except Exception as e:
            self.log_security_event("EXECUTION_ERROR", {"error": str(e)})
            raise
        finally:
            self.logger.info("Secure execution completed")

    def cleanup(self):
        """Cleanup resources on shutdown"""
        self.logger.info("Performing security cleanup")
        # Clear sensitive data from memory
        # Close database connections
        # Cleanup temporary files

# Global security manager
security_manager = SecurityManager()

async def run_secure_execution(inputs: dict) -> dict:
    """Execute with security monitoring"""
    with security_manager.secure_execution():
        # Log request
        security_manager.logger.info(f"Processing request: {inputs.get('request_id', 'unknown')}")

        # Process request
        result = await process_secure_request(inputs)

        return result
```

## Network Security

### TLS/SSL Configuration

Secure network communications:

```python
import ssl
import aiohttp
from typing import Optional

class SecureHTTPClient:
    def __init__(self):
        # Create secure SSL context
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = True
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED

        # Additional security settings
        self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

    async def make_secure_request(self, url: str, data: dict = None, headers: dict = None) -> dict:
        """Make secure HTTPS request"""
        default_headers = {
            'User-Agent': 'Chutes-Secure-Client/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        if headers:
            default_headers.update(headers)

        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(ssl=self.ssl_context)
        ) as session:
            async with session.post(url, json=data, headers=default_headers) as response:
                if response.status != 200:
                    raise Exception(f"Request failed: {response.status}")

                return await response.json()

# Certificate pinning for critical services
class CertificatePinnedClient:
    def __init__(self, pinned_cert_fingerprint: str):
        self.pinned_fingerprint = pinned_cert_fingerprint

    def verify_certificate(self, cert_der: bytes) -> bool:
        """Verify certificate against pinned fingerprint"""
        import hashlib
        cert_fingerprint = hashlib.sha256(cert_der).hexdigest()
        return cert_fingerprint == self.pinned_fingerprint

    async def make_pinned_request(self, url: str, data: dict) -> dict:
        """Make request with certificate pinning"""
        # Implementation would verify certificate fingerprint
        # This is a simplified example
        client = SecureHTTPClient()
        return await client.make_secure_request(url, data)
```

### Rate Limiting

Implement rate limiting:

```python
import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Optional

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour

        # Track requests per client
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        current_time = time.time()

        # Clean old requests
        self._cleanup_old_requests(client_id, current_time)

        # Check limits
        minute_count = len(self.minute_requests[client_id])
        hour_count = len(self.hour_requests[client_id])

        if minute_count >= self.rpm_limit or hour_count >= self.rph_limit:
            return False

        # Record request
        self.minute_requests[client_id].append(current_time)
        self.hour_requests[client_id].append(current_time)

        return True

    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Remove old requests from tracking"""
        minute_cutoff = current_time - 60  # 1 minute ago
        hour_cutoff = current_time - 3600  # 1 hour ago

        # Clean minute requests
        while (self.minute_requests[client_id] and
               self.minute_requests[client_id][0] < minute_cutoff):
            self.minute_requests[client_id].popleft()

        # Clean hour requests
        while (self.hour_requests[client_id] and
               self.hour_requests[client_id][0] < hour_cutoff):
            self.hour_requests[client_id].popleft()

    def get_reset_time(self, client_id: str) -> Dict[str, int]:
        """Get time until rate limit resets"""
        current_time = time.time()

        next_minute_reset = 60 - (current_time % 60)
        next_hour_reset = 3600 - (current_time % 3600)

        return {
            "minute_reset": int(next_minute_reset),
            "hour_reset": int(next_hour_reset)
        }

# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=100, requests_per_hour=5000)

async def run_with_rate_limiting(inputs: dict) -> dict:
    """Apply rate limiting to requests"""
    client_id = inputs.get("client_id") or inputs.get("user_id", "unknown")

    if not rate_limiter.is_allowed(client_id):
        reset_times = rate_limiter.get_reset_time(client_id)
        return {
            "error": "Rate limit exceeded",
            "status": 429,
            "reset_time": reset_times
        }

    # Process request
    result = await process_rate_limited_request(inputs)
    return result
```

## Monitoring & Incident Response

### Security Monitoring

Monitor for security threats:

```python
import logging
import time
from collections import defaultdict
from typing import Dict, List
import json

class SecurityMonitor:
    def __init__(self):
        self.logger = logging.getLogger('security_monitor')

        # Track suspicious activities
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)

        # Threat detection thresholds
        self.max_failed_attempts = 5
        self.time_window = 300  # 5 minutes
        self.alert_threshold = 10

    def log_failed_authentication(self, client_id: str, details: dict):
        """Log failed authentication attempt"""
        current_time = time.time()
        self.failed_attempts[client_id].append(current_time)

        # Clean old attempts
        cutoff_time = current_time - self.time_window
        self.failed_attempts[client_id] = [
            t for t in self.failed_attempts[client_id] if t > cutoff_time
        ]

        # Check for brute force attack
        if len(self.failed_attempts[client_id]) >= self.max_failed_attempts:
            self.alert_brute_force_attack(client_id, details)

    def alert_brute_force_attack(self, client_id: str, details: dict):
        """Alert on potential brute force attack"""
        alert = {
            "alert_type": "BRUTE_FORCE_ATTACK",
            "client_id": client_id,
            "attempt_count": len(self.failed_attempts[client_id]),
            "time_window": self.time_window,
            "details": details,
            "timestamp": time.time()
        }

        self.logger.critical(f"SECURITY ALERT: {json.dumps(alert)}")

        # In production, send to SIEM or alerting system
        self.send_security_alert(alert)

    def detect_suspicious_patterns(self, request_data: dict) -> bool:
        """Detect suspicious request patterns"""
        suspicious_indicators = [
            # SQL injection patterns
            r'(\bUNION\b.*\bSELECT\b)|(\bSELECT\b.*\bFROM\b)',
            # XSS patterns
            r'<script|javascript:|onload=|onerror=',
            # Path traversal
            r'\.\./|\.\.\\'
        ]

        request_str = json.dumps(request_data).lower()

        for pattern in suspicious_indicators:
            if re.search(pattern, request_str, re.IGNORECASE):
                self.log_suspicious_activity("INJECTION_ATTEMPT", {
                    "pattern": pattern,
                    "request": request_data
                })
                return True

        return False

    def log_suspicious_activity(self, activity_type: str, details: dict):
        """Log suspicious activity"""
        self.suspicious_patterns[activity_type] += 1

        if self.suspicious_patterns[activity_type] >= self.alert_threshold:
            self.alert_suspicious_pattern(activity_type, details)

    def alert_suspicious_pattern(self, pattern_type: str, details: dict):
        """Alert on suspicious activity pattern"""
        alert = {
            "alert_type": "SUSPICIOUS_PATTERN",
            "pattern_type": pattern_type,
            "occurrence_count": self.suspicious_patterns[pattern_type],
            "details": details,
            "timestamp": time.time()
        }

        self.logger.critical(f"SECURITY ALERT: {json.dumps(alert)}")
        self.send_security_alert(alert)

    def send_security_alert(self, alert: dict):
        """Send security alert to monitoring system"""
        # In production, integrate with:
        # - SIEM systems (Splunk, ELK Stack)
        # - Alerting platforms (PagerDuty, Slack)
        # - Security orchestration tools
        pass

# Global security monitor
security_monitor = SecurityMonitor()

async def run_with_security_monitoring(inputs: dict) -> dict:
    """Monitor requests for security threats"""
    client_id = inputs.get("client_id", "unknown")

    # Check for suspicious patterns
    if security_monitor.detect_suspicious_patterns(inputs):
        return {"error": "Suspicious request blocked", "status": 403}

    try:
        # Process request
        result = await process_monitored_request(inputs)
        return result

    except Exception as e:
        # Log potential security incident
        security_monitor.log_suspicious_activity("REQUEST_ERROR", {
            "error": str(e),
            "client_id": client_id,
            "inputs": inputs
        })
        raise
```

### Incident Response

Automated incident response:

```python
import asyncio
from enum import Enum
from typing import Dict, List, Callable

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class IncidentResponse:
    def __init__(self):
        self.response_handlers: Dict[str, Callable] = {}
        self.blocked_clients: set = set()
        self.incident_log: List[dict] = []

    def register_handler(self, incident_type: str, handler: Callable):
        """Register incident response handler"""
        self.response_handlers[incident_type] = handler

    async def handle_incident(self, incident_type: str, severity: IncidentSeverity, details: dict):
        """Handle security incident"""
        incident = {
            "type": incident_type,
            "severity": severity.name,
            "details": details,
            "timestamp": time.time(),
            "status": "ACTIVE"
        }

        self.incident_log.append(incident)

        # Execute response handler
        if incident_type in self.response_handlers:
            await self.response_handlers[incident_type](incident)

        # Default responses based on severity
        if severity == IncidentSeverity.CRITICAL:
            await self.emergency_response(incident)
        elif severity == IncidentSeverity.HIGH:
            await self.high_priority_response(incident)

    async def emergency_response(self, incident: dict):
        """Emergency response for critical incidents"""
        client_id = incident["details"].get("client_id")

        # Immediately block client
        if client_id:
            self.blocked_clients.add(client_id)

        # Notify security team
        await self.notify_security_team(incident)

        # Scale down if under attack
        await self.initiate_defensive_scaling()

    async def high_priority_response(self, incident: dict):
        """High priority incident response"""
        client_id = incident["details"].get("client_id")

        # Temporarily throttle client
        if client_id:
            await self.throttle_client(client_id)

        # Alert monitoring systems
        await self.send_alert(incident)

    async def notify_security_team(self, incident: dict):
        """Notify security team of critical incident"""
        # Integration with alerting systems
        pass

    async def initiate_defensive_scaling(self):
        """Scale resources defensively during attack"""
        # Implement defensive scaling logic
        pass

    async def throttle_client(self, client_id: str):
        """Apply temporary throttling to client"""
        # Implement client throttling
        pass

    def is_client_blocked(self, client_id: str) -> bool:
        """Check if client is blocked"""
        return client_id in self.blocked_clients

# Global incident response
incident_response = IncidentResponse()

# Register handlers
async def brute_force_handler(incident: dict):
    """Handle brute force attack"""
    client_id = incident["details"].get("client_id")
    if client_id:
        incident_response.blocked_clients.add(client_id)

incident_response.register_handler("BRUTE_FORCE_ATTACK", brute_force_handler)

async def run_with_incident_response(inputs: dict) -> dict:
    """Process requests with incident response"""
    client_id = inputs.get("client_id", "unknown")

    # Check if client is blocked
    if incident_response.is_client_blocked(client_id):
        return {"error": "Client blocked due to security incident", "status": 403}

    # Process request
    result = await process_secure_request(inputs)
    return result
```

## Security Checklist

### Pre-deployment Security

- [ ] Enable authentication and authorization
- [ ] Implement input validation and sanitization
- [ ] Use encryption for sensitive data
- [ ] Build secure Docker images
- [ ] Configure TLS/SSL properly
- [ ] Set up rate limiting
- [ ] Implement security monitoring
- [ ] Test for common vulnerabilities

### Runtime Security

- [ ] Monitor for security events
- [ ] Implement incident response procedures
- [ ] Keep dependencies updated
- [ ] Regular security audits
- [ ] Backup and recovery procedures
- [ ] Access logging and monitoring

### Compliance Considerations

- [ ] GDPR compliance for EU users
- [ ] HIPAA compliance for healthcare data
- [ ] SOC 2 compliance for enterprise customers
- [ ] Industry-specific security requirements

## Next Steps

- **[Best Practices](best-practices)** - General security best practices
- **[Compliance Guide](../compliance)** - Meet regulatory requirements
- **[Monitoring](../monitoring)** - Advanced security monitoring
- **[Incident Response Playbook](../incident-response)** - Detailed response procedures

For enterprise security requirements, see the [Enterprise Security Guide](../enterprise/security).
