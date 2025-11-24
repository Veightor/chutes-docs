# Nodes API Reference

This section covers all endpoints related to nodes.


## List Nodes

List full inventory, optionally in detailed view (which lists chutes).


<div class="api-test-widget" data-widget-id="widget_get__nodes_"></div>
<script type="application/json" data-widget-config="widget_get__nodes_">{"endpoint":"/nodes/","method":"GET","parameters":[{"name":"model","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"detailed","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"hotkey","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /nodes/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string \| null | No |  |
| detailed | boolean \| null | No |  |
| hotkey | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Create Nodes

Add nodes/GPUs to inventory.


<div class="api-test-widget" data-widget-id="widget_post__nodes_"></div>
<script type="application/json" data-widget-config="widget_post__nodes_">{"endpoint":"/nodes/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"server_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Server Id"},"nodes":{"items":{"$ref":"#/components/schemas/NodeArgs"},"type":"array","title":"Nodes"}},"required":["nodes"]}}</script>

**Endpoint:** `POST /nodes/`

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
| server_id | string \| null | No |  |
| nodes | NodeArgs[] | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 202 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Supported Gpus

Show all currently supported GPUs.


<div class="api-test-widget" data-widget-id="widget_get__nodes_supported"></div>
<script type="application/json" data-widget-config="widget_get__nodes_supported">{"endpoint":"/nodes/supported","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /nodes/supported`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Check Verification Status

Check taskiq task status, to see if the validator has finished GPU verification.


<div class="api-test-widget" data-widget-id="widget_get__nodes_verification_status"></div>
<script type="application/json" data-widget-config="widget_get__nodes_verification_status">{"endpoint":"/nodes/verification_status","method":"GET","requiresAuth":true,"parameters":[{"name":"task_id","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /nodes/verification_status`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| task_id | string | Yes |  |
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

## Delete Node

Remove a node from inventory.


<div class="api-test-widget" data-widget-id="widget_delete__nodes__node_id_"></div>
<script type="application/json" data-widget-config="widget_delete__nodes__node_id_">{"endpoint":"/nodes/{node_id}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"node_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /nodes/{node_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| node_id | string | Yes |  |
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
