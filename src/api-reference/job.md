# Job API Reference

This section covers all endpoints related to job.


## Create Job

Create a job.


<div class="api-test-widget" data-widget-id="widget_post__jobs__chute_id___method_"></div>
<script type="application/json" data-widget-config="widget_post__jobs__chute_id___method_">{"endpoint":"/jobs/{chute_id}/{method}","method":"POST","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"method","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /jobs/{chute_id}/{method}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| method | string | Yes |  |
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

## Delete Job

Delete a job.


<div class="api-test-widget" data-widget-id="widget_delete__jobs__job_id_"></div>
<script type="application/json" data-widget-config="widget_delete__jobs__job_id_">{"endpoint":"/jobs/{job_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /jobs/{job_id}`

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

## Finish Job And Get Upload Targets

Mark a job as complete (which could be failed; "done" either way)


<div class="api-test-widget" data-widget-id="widget_post__jobs__job_id_"></div>
<script type="application/json" data-widget-config="widget_post__jobs__job_id_">{"endpoint":"/jobs/{job_id}","method":"POST","parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"token","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `POST /jobs/{job_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string | Yes |  |
| token | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Complete Job

Final update, which checks the file uploads to see which were successfully transferred etc.


<div class="api-test-widget" data-widget-id="widget_put__jobs__job_id_"></div>
<script type="application/json" data-widget-config="widget_put__jobs__job_id_">{"endpoint":"/jobs/{job_id}","method":"PUT","parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"token","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `PUT /jobs/{job_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string | Yes |  |
| token | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Job

Get a job.


<div class="api-test-widget" data-widget-id="widget_get__jobs__job_id_"></div>
<script type="application/json" data-widget-config="widget_get__jobs__job_id_">{"endpoint":"/jobs/{job_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /jobs/{job_id}`

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

## Upload Job File

Upload a job's output file.


<div class="api-test-widget" data-widget-id="widget_put__jobs__job_id__upload"></div>
<script type="application/json" data-widget-config="widget_put__jobs__job_id__upload">{"endpoint":"/jobs/{job_id}/upload","method":"PUT","parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"token","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `PUT /jobs/{job_id}/upload`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string | Yes |  |
| token | string | Yes |  |



### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Download Output File

Download a job's output file.


<div class="api-test-widget" data-widget-id="widget_get__jobs__job_id__download__file_id_"></div>
<script type="application/json" data-widget-config="widget_get__jobs__job_id__download__file_id_">{"endpoint":"/jobs/{job_id}/download/{file_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"job_id","type":"string","required":true,"description":"","in":"path"},{"name":"file_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /jobs/{job_id}/download/{file_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string | Yes |  |
| file_id | string | Yes |  |
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
