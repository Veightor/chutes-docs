# Audit API Reference

This section covers all endpoints related to audit.


## Add Miner Audit Data


<div class="api-test-widget" data-widget-id="widget_post__audit_miner_data"></div>
<script type="application/json" data-widget-config="widget_post__audit_miner_data">{"endpoint":"/audit/miner_data","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Block","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /audit/miner_data`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Block | string \| null | No |  |
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

## List Audit Entries

List all audit reports from the past week.


<div class="api-test-widget" data-widget-id="widget_get__audit_"></div>
<script type="application/json" data-widget-config="widget_get__audit_">{"endpoint":"/audit/","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /audit/`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Download Audit Data

Download report data.


<div class="api-test-widget" data-widget-id="widget_get__audit_download"></div>
<script type="application/json" data-widget-config="widget_get__audit_download">{"endpoint":"/audit/download","method":"GET","parameters":[{"name":"path","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /audit/download`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| path | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
