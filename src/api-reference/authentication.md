# Authentication API Reference

This section covers all endpoints related to authentication.


## Registry Auth

Authentication registry/docker pull requests.


<div class="api-test-widget" data-widget-id="widget_get__registry_auth"></div>
<script type="application/json" data-widget-config="widget_get__registry_auth">{"endpoint":"/registry/auth","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /registry/auth`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Keys

List (and optionally filter/paginate) keys.


<div class="api-test-widget" data-widget-id="widget_get__api_keys_"></div>
<script type="application/json" data-widget-config="widget_get__api_keys_">{"endpoint":"/api_keys/","method":"GET","requiresAuth":true,"parameters":[{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /api_keys/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Create Api Key

Create a new API key.


<div class="api-test-widget" data-widget-id="widget_post__api_keys_"></div>
<script type="application/json" data-widget-config="widget_post__api_keys_">{"endpoint":"/api_keys/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"admin":{"type":"boolean","title":"Admin"},"name":{"type":"string","title":"Name"},"scopes":{"anyOf":[{"items":{"$ref":"#/components/schemas/ScopeArgs"},"type":"array"},{"type":"null"}],"title":"Scopes","default":[]}},"required":["admin","name"]}}</script>

**Endpoint:** `POST /api_keys/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| admin | boolean | Yes |  |
| name | string | Yes |  |
| scopes | ScopeArgs[] \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Key

Get a single key.


<div class="api-test-widget" data-widget-id="widget_get__api_keys__api_key_id_"></div>
<script type="application/json" data-widget-config="widget_get__api_keys__api_key_id_">{"endpoint":"/api_keys/{api_key_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"api_key_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /api_keys/{api_key_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| api_key_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Delete Api Key

Delete an API key by ID.


<div class="api-test-widget" data-widget-id="widget_delete__api_keys__api_key_id_"></div>
<script type="application/json" data-widget-config="widget_delete__api_keys__api_key_id_">{"endpoint":"/api_keys/{api_key_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"api_key_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /api_keys/{api_key_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| api_key_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---
