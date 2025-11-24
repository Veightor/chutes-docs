# API Overview

Welcome to the Chutes API documentation. Our REST API provides programmatic access to all Chutes platform functionality, allowing you to integrate Chutes into your applications and workflows.

## Getting Started

The Chutes API is organized around REST principles. Our API has predictable resource-oriented URLs, accepts form-encoded request bodies, returns JSON-encoded responses, and uses standard HTTP response codes, authentication, and verbs.

### Base URL

All API endpoints are relative to the base URL:

```
https://api.chutes.ai
```

### Authentication

The Chutes API uses API key authentication. Include your API key in the `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.chutes.ai/endpoint
```

You can obtain your API key from the [Chutes dashboard](https://chutes.ai/app).

## Core Resources

The Chutes API is built around several core resources:

## Available APIs

### [Users](users)

20 endpoints

### [Chutes](chutes)

18 endpoints

### [Images](images)

5 endpoints

### [Nodes](nodes)

5 endpoints

### [Pricing](pricing)

6 endpoints

### [Instances](instances)

8 endpoints

### [Invocations](invocations)

6 endpoints

### [Authentication](authentication)

5 endpoints

### [Miner](miner)

15 endpoints

### [Logo](logo)

2 endpoints

### [Configguesser](configguesser)

1 endpoint

### [Audit](audit)

3 endpoints

### [Job](job)

7 endpoints

### [General](general)

2 endpoints
