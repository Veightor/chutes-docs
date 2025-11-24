# Invocations API Reference

This section covers all endpoints related to invocations.


## Get Usage

Get aggregated usage data, which is the amount of revenue
we would be receiving if no usage was free.


<div class="api-test-widget" data-widget-id="widget_get__invocations_usage"></div>
<script type="application/json" data-widget-config="widget_get__invocations_usage">{"endpoint":"/invocations/usage","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /invocations/usage`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Llm Stats


<div class="api-test-widget" data-widget-id="widget_get__invocations_stats_llm"></div>
<script type="application/json" data-widget-config="widget_get__invocations_stats_llm">{"endpoint":"/invocations/stats/llm","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /invocations/stats/llm`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Diffusion Stats


<div class="api-test-widget" data-widget-id="widget_get__invocations_stats_diffusion"></div>
<script type="application/json" data-widget-config="widget_get__invocations_stats_diffusion">{"endpoint":"/invocations/stats/diffusion","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /invocations/stats/diffusion`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Export

Get invocation exports (and reports) for a particular hour.


<div class="api-test-widget" data-widget-id="widget_get__invocations_exports__year___month___day___hour_format_"></div>
<script type="application/json" data-widget-config="widget_get__invocations_exports__year___month___day___hour_format_">{"endpoint":"/invocations/exports/{year}/{month}/{day}/{hour_format}","method":"GET","parameters":[{"name":"year","type":"integer","required":true,"description":"","in":"path"},{"name":"month","type":"integer","required":true,"description":"","in":"path"},{"name":"day","type":"integer","required":true,"description":"","in":"path"},{"name":"hour_format","type":"string","required":true,"description":"","in":"path"}],"requestBody":null}</script>

**Endpoint:** `GET /invocations/exports/{year}/{month}/{day}/{hour_format}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| year | integer | Yes |  |
| month | integer | Yes |  |
| day | integer | Yes |  |
| hour_format | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Recent Export

Get an export for recent data, which may not yet be in S3.


<div class="api-test-widget" data-widget-id="widget_get__invocations_exports_recent"></div>
<script type="application/json" data-widget-config="widget_get__invocations_exports_recent">{"endpoint":"/invocations/exports/recent","method":"GET","parameters":[{"name":"hotkey","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /invocations/exports/recent`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| hotkey | string \| null | No |  |
| limit | integer \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Report Invocation


<div class="api-test-widget" data-widget-id="widget_post__invocations__invocation_id__report"></div>
<script type="application/json" data-widget-config="widget_post__invocations__invocation_id__report">{"endpoint":"/invocations/{invocation_id}/report","method":"POST","requiresAuth":true,"parameters":[{"name":"invocation_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"reason":{"type":"string","title":"Reason"}},"required":["reason"]}}</script>

**Endpoint:** `POST /invocations/{invocation_id}/report`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| invocation_id | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| reason | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---
