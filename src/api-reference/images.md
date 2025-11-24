# Images API Reference

This section covers all endpoints related to images.


## Stream Build Logs


<div class="api-test-widget" data-widget-id="widget_get__images__image_id__logs"></div>
<script type="application/json" data-widget-config="widget_get__images__image_id__logs">{"endpoint":"/images/{image_id}/logs","method":"GET","requiresAuth":true,"parameters":[{"name":"image_id","type":"string","required":true,"description":"","in":"path"},{"name":"offset","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /images/{image_id}/logs`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_id | string | Yes |  |
| offset | string \| null | No |  |
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

List (and optionally filter/paginate) images.


<div class="api-test-widget" data-widget-id="widget_get__images_"></div>
<script type="application/json" data-widget-config="widget_get__images_">{"endpoint":"/images/","method":"GET","requiresAuth":true,"parameters":[{"name":"include_public","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"name","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"tag","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /images/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| include_public | boolean \| null | No |  |
| name | string \| null | No |  |
| tag | string \| null | No |  |
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

## Create Image

Create an image; really here we're just storing the metadata
in the DB and kicking off the image build asynchronously.


<div class="api-test-widget" data-widget-id="widget_post__images_"></div>
<script type="application/json" data-widget-config="widget_post__images_">{"endpoint":"/images/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /images/`

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
| 202 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Get Image

Load a single image by ID or name.


<div class="api-test-widget" data-widget-id="widget_get__images__image_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_get__images__image_id_or_name_">{"endpoint":"/images/{image_id_or_name}","method":"GET","requiresAuth":true,"parameters":[{"name":"image_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /images/{image_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_id_or_name | string | Yes |  |
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

## Delete Image

Delete an image by ID or name:tag.


<div class="api-test-widget" data-widget-id="widget_delete__images__image_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_delete__images__image_id_or_name_">{"endpoint":"/images/{image_id_or_name}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"image_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /images/{image_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_id_or_name | string | Yes |  |
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
