# Logo API Reference

This section covers all endpoints related to logo.


## Create Logo

Create/upload a new logo.


<div class="api-test-widget" data-widget-id="widget_post__logos_"></div>
<script type="application/json" data-widget-config="widget_post__logos_">{"endpoint":"/logos/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /logos/`

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

## Render Logo

Logo image response.


<div class="api-test-widget" data-widget-id="widget_get__logos__logo_id___extension_"></div>
<script type="application/json" data-widget-config="widget_get__logos__logo_id___extension_">{"endpoint":"/logos/{logo_id}.{extension}","method":"GET","parameters":[{"name":"logo_id","type":"string","required":true,"description":"","in":"path"},{"name":"extension","type":"string","required":true,"description":"","in":"path"}],"requestBody":null}</script>

**Endpoint:** `GET /logos/{logo_id}.{extension}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| logo_id | string | Yes |  |
| extension | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
