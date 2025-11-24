# Miner API Reference

This section covers all endpoints related to miner.


## List Chutes


<div class="api-test-widget" data-widget-id="widget_get__miner_chutes_"></div>
<script type="application/json" data-widget-config="widget_get__miner_chutes_">{"endpoint":"/miner/chutes/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/chutes/`

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

## List Images


<div class="api-test-widget" data-widget-id="widget_get__miner_images_"></div>
<script type="application/json" data-widget-config="widget_get__miner_images_">{"endpoint":"/miner/images/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/images/`

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

## List Nodes


<div class="api-test-widget" data-widget-id="widget_get__miner_nodes_"></div>
<script type="application/json" data-widget-config="widget_get__miner_nodes_">{"endpoint":"/miner/nodes/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/nodes/`

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

## List Instances


<div class="api-test-widget" data-widget-id="widget_get__miner_instances_"></div>
<script type="application/json" data-widget-config="widget_get__miner_instances_">{"endpoint":"/miner/instances/","method":"GET","requiresAuth":true,"parameters":[{"name":"explicit_null","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/instances/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| explicit_null | boolean \| null | No |  |
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

## List Available Jobs


<div class="api-test-widget" data-widget-id="widget_get__miner_jobs_"></div>
<script type="application/json" data-widget-config="widget_get__miner_jobs_">{"endpoint":"/miner/jobs/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/jobs/`

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

## Release Job


<div class="api-test-widget" data-widget-id="widget_delete__miner_jobs__job_id_"></div>
<script type="application/json" data-widget-config="widget_delete__miner_jobs__job_id_">{"endpoint":"/miner/jobs/{job_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /miner/jobs/{job_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string | Yes |  |
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

## Get Full Inventory


<div class="api-test-widget" data-widget-id="widget_get__miner_inventory"></div>
<script type="application/json" data-widget-config="widget_get__miner_inventory">{"endpoint":"/miner/inventory","method":"GET","parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/inventory`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Metrics


<div class="api-test-widget" data-widget-id="widget_get__miner_metrics_"></div>
<script type="application/json" data-widget-config="widget_get__miner_metrics_">{"endpoint":"/miner/metrics/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/metrics/`

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

## Get Chute


<div class="api-test-widget" data-widget-id="widget_get__miner_chutes__chute_id___version_"></div>
<script type="application/json" data-widget-config="widget_get__miner_chutes__chute_id___version_">{"endpoint":"/miner/chutes/{chute_id}/{version}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"version","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/chutes/{chute_id}/{version}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| version | string | Yes |  |
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

## Get Stats

Get invocation status over different intervals.


<div class="api-test-widget" data-widget-id="widget_get__miner_stats"></div>
<script type="application/json" data-widget-config="widget_get__miner_stats">{"endpoint":"/miner/stats","method":"GET","parameters":[{"name":"miner_hotkey","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"per_chute","type":"boolean \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/stats`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| miner_hotkey | string \| null | No |  |
| per_chute | boolean \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Scores


<div class="api-test-widget" data-widget-id="widget_get__miner_scores"></div>
<script type="application/json" data-widget-config="widget_get__miner_scores">{"endpoint":"/miner/scores","method":"GET","parameters":[{"name":"hotkey","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/scores`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| hotkey | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Unique Chute History


<div class="api-test-widget" data-widget-id="widget_get__miner_unique_chute_history__hotkey_"></div>
<script type="application/json" data-widget-config="widget_get__miner_unique_chute_history__hotkey_">{"endpoint":"/miner/unique_chute_history/{hotkey}","method":"GET","parameters":[{"name":"hotkey","type":"string","required":true,"description":"","in":"path"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/unique_chute_history/{hotkey}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| hotkey | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Metagraph


<div class="api-test-widget" data-widget-id="widget_get__miner_metagraph"></div>
<script type="application/json" data-widget-config="widget_get__miner_metagraph">{"endpoint":"/miner/metagraph","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /miner/metagraph`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Utilization


<div class="api-test-widget" data-widget-id="widget_get__miner_utilization"></div>
<script type="application/json" data-widget-config="widget_get__miner_utilization">{"endpoint":"/miner/utilization","method":"GET","parameters":[{"name":"hotkey","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/utilization`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| hotkey | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Utilization Instances


<div class="api-test-widget" data-widget-id="widget_get__miner_utilization_instances"></div>
<script type="application/json" data-widget-config="widget_get__miner_utilization_instances">{"endpoint":"/miner/utilization_instances","method":"GET","parameters":[{"name":"hotkey","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /miner/utilization_instances`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| hotkey | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
