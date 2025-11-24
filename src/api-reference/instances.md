# Instances API Reference

This section covers all endpoints related to instances.


## Get Launch Config


<div class="api-test-widget" data-widget-id="widget_get__instances_launch_config"></div>
<script type="application/json" data-widget-config="widget_get__instances_launch_config">{"endpoint":"/instances/launch_config","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"query"},{"name":"job_id","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/launch_config`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| job_id | string \| null | No |  |
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

## Claim Launch Config


<div class="api-test-widget" data-widget-id="widget_post__instances_launch_config__config_id_"></div>
<script type="application/json" data-widget-config="widget_post__instances_launch_config__config_id_">{"endpoint":"/instances/launch_config/{config_id}","method":"POST","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"gpus":{"items":{"type":"object"},"type":"array","title":"Gpus"},"host":{"type":"string","title":"Host"},"port_mappings":{"items":{"$ref":"#/components/schemas/PortMap"},"type":"array","title":"Port Mappings"},"env":{"type":"string","title":"Env"},"code":{"type":"string","title":"Code"},"fsv":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Fsv"}},"required":["gpus","host","port_mappings","env","code"]}}</script>

**Endpoint:** `POST /instances/launch_config/{config_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| gpus | object[] | Yes |  |
| host | string | Yes |  |
| port_mappings | PortMap[] | Yes |  |
| env | string | Yes |  |
| code | string | Yes |  |
| fsv | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Verify Launch Config Instance


<div class="api-test-widget" data-widget-id="widget_put__instances_launch_config__config_id_"></div>
<script type="application/json" data-widget-config="widget_put__instances_launch_config__config_id_">{"endpoint":"/instances/launch_config/{config_id}","method":"PUT","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /instances/launch_config/{config_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Activate Launch Config Instance


<div class="api-test-widget" data-widget-id="widget_get__instances_launch_config__config_id__activate"></div>
<script type="application/json" data-widget-config="widget_get__instances_launch_config__config_id__activate">{"endpoint":"/instances/launch_config/{config_id}/activate","method":"GET","parameters":[{"name":"config_id","type":"string","required":true,"description":"","in":"path"},{"name":"Authorization","type":"string","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/launch_config/{config_id}/activate`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| config_id | string | Yes |  |
| Authorization | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Create Instance


<div class="api-test-widget" data-widget-id="widget_post__instances__chute_id__"></div>
<script type="application/json" data-widget-config="widget_post__instances__chute_id__">{"endpoint":"/instances/{chute_id}/","method":"POST","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"node_ids":{"items":{"type":"string"},"type":"array","title":"Node Ids"},"host":{"type":"string","title":"Host"},"port":{"type":"integer","title":"Port"}},"required":["node_ids","host","port"]}}</script>

**Endpoint:** `POST /instances/{chute_id}/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| node_ids | string[] | Yes |  |
| host | string | Yes |  |
| port | integer | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 202 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Token


<div class="api-test-widget" data-widget-id="widget_get__instances_token_check"></div>
<script type="application/json" data-widget-config="widget_get__instances_token_check">{"endpoint":"/instances/token_check","method":"GET","parameters":[{"name":"salt","type":"string","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /instances/token_check`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| salt | string | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Activate Instance


<div class="api-test-widget" data-widget-id="widget_patch__instances__chute_id___instance_id_"></div>
<script type="application/json" data-widget-config="widget_patch__instances__chute_id___instance_id_">{"endpoint":"/instances/{chute_id}/{instance_id}","method":"PATCH","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"instance_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"active":{"type":"boolean","title":"Active"}},"required":["active"]}}</script>

**Endpoint:** `PATCH /instances/{chute_id}/{instance_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| instance_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| active | boolean | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Delete Instance


<div class="api-test-widget" data-widget-id="widget_delete__instances__chute_id___instance_id_"></div>
<script type="application/json" data-widget-config="widget_delete__instances__chute_id___instance_id_">{"endpoint":"/instances/{chute_id}/{instance_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"instance_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /instances/{chute_id}/{instance_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| instance_id | string | Yes |  |
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
