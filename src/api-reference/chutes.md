# Chutes API Reference

This section covers all endpoints related to chutes.


## List Chutes

List (and optionally filter/paginate) chutes.


<div class="api-test-widget" data-widget-id="widget_get__chutes_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_">{"endpoint":"/chutes/","method":"GET","requiresAuth":true,"parameters":[{"name":"include_public","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"template","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"name","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"image","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"slug","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"include_schemas","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| include_public | boolean \| null | No |  |
| template | string \| null | No |  |
| name | string \| null | No |  |
| image | string \| null | No |  |
| slug | string \| null | No |  |
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
| include_schemas | boolean \| null | No |  |
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

## Deploy Chute

Standard deploy from the CDK.


<div class="api-test-widget" data-widget-id="widget_post__chutes_"></div>
<script type="application/json" data-widget-config="widget_post__chutes_">{"endpoint":"/chutes/","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"name":{"type":"string","maxLength":128,"minLength":3,"title":"Name"},"tagline":{"anyOf":[{"type":"string","maxLength":1024},{"type":"null"}],"title":"Tagline","default":""},"readme":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Readme","default":""},"tool_description":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Tool Description"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"image":{"type":"string","title":"Image"},"public":{"type":"boolean","title":"Public"},"code":{"type":"string","title":"Code"},"filename":{"type":"string","title":"Filename"},"ref_str":{"type":"string","title":"Ref Str"},"standard_template":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Standard Template"},"node_selector":{"$ref":"#/components/schemas/NodeSelector"},"cords":{"anyOf":[{"items":{"$ref":"#/components/schemas/Cord"},"type":"array"},{"type":"null"}],"title":"Cords","default":[]},"jobs":{"anyOf":[{"items":{"$ref":"#/components/schemas/Job"},"type":"array"},{"type":"null"}],"title":"Jobs","default":[]},"concurrency":{"anyOf":[{"type":"integer","maximum":256},{"type":"null"}],"title":"Concurrency","gte":0},"revision":{"anyOf":[{"type":"string","pattern":"^[a-fA-F0-9]{40}$"},{"type":"null"}],"title":"Revision"}},"required":["name","image","public","code","filename","ref_str","node_selector"]}}</script>

**Endpoint:** `POST /chutes/`

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
| name | string | Yes |  |
| tagline | string \| null | No |  |
| readme | string \| null | No |  |
| tool_description | string \| null | No |  |
| logo_id | string \| null | No |  |
| image | string | Yes |  |
| public | boolean | Yes |  |
| code | string | Yes |  |
| filename | string | Yes |  |
| ref_str | string | Yes |  |
| standard_template | string \| null | No |  |
| node_selector | NodeSelector | Yes |  |
| cords | Cord[] \| null | No |  |
| jobs | Job[] \| null | No |  |
| concurrency | integer \| null | No |  |
| revision | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Rolling Updates


<div class="api-test-widget" data-widget-id="widget_get__chutes_rolling_updates"></div>
<script type="application/json" data-widget-config="widget_get__chutes_rolling_updates">{"endpoint":"/chutes/rolling_updates","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/rolling_updates`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Gpu Count History


<div class="api-test-widget" data-widget-id="widget_get__chutes_gpu_count_history"></div>
<script type="application/json" data-widget-config="widget_get__chutes_gpu_count_history">{"endpoint":"/chutes/gpu_count_history","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/gpu_count_history`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Chute Miner Mean Index


<div class="api-test-widget" data-widget-id="widget_get__chutes_miner_means"></div>
<script type="application/json" data-widget-config="widget_get__chutes_miner_means">{"endpoint":"/chutes/miner_means","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/miner_means`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Chute Miner Means

Load a chute's mean TPS and output token count by miner ID.


<div class="api-test-widget" data-widget-id="widget_get__chutes_miner_means__chute_id___ext_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_miner_means__chute_id___ext_">{"endpoint":"/chutes/miner_means/{chute_id}.{ext}","method":"GET","parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"ext","type":"string \\| null","required":true,"description":"","in":"path"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/miner_means/{chute_id}.{ext}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| ext | string \| null | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Chute Miner Means

Load a chute's mean TPS and output token count by miner ID.


<div class="api-test-widget" data-widget-id="widget_get__chutes_miner_means__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_miner_means__chute_id_">{"endpoint":"/chutes/miner_means/{chute_id}","method":"GET","parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"ext","type":"string \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/miner_means/{chute_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
| ext | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Get Chute Code

Load a chute's code by ID or name.


<div class="api-test-widget" data-widget-id="widget_get__chutes_code__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__chutes_code__chute_id_">{"endpoint":"/chutes/code/{chute_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/code/{chute_id}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
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

## Get Chute Utilization

Get chute utilization data.


<div class="api-test-widget" data-widget-id="widget_get__chutes_utilization_legacy"></div>
<script type="application/json" data-widget-config="widget_get__chutes_utilization_legacy">{"endpoint":"/chutes/utilization_legacy","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/utilization_legacy`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Chute Utilization V2

Get chute utilization data from the most recent capacity log.


<div class="api-test-widget" data-widget-id="widget_get__chutes_utilization"></div>
<script type="application/json" data-widget-config="widget_get__chutes_utilization">{"endpoint":"/chutes/utilization","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /chutes/utilization`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Chute

Load a chute by ID or name.


<div class="api-test-widget" data-widget-id="widget_get__chutes__chute_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_get__chutes__chute_id_or_name_">{"endpoint":"/chutes/{chute_id_or_name}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /chutes/{chute_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
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

## Delete Chute

Delete a chute by ID or name.


<div class="api-test-widget" data-widget-id="widget_delete__chutes__chute_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_delete__chutes__chute_id_or_name_">{"endpoint":"/chutes/{chute_id_or_name}","method":"DELETE","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /chutes/{chute_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
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

## Update Common Attributes

Update readme, tagline, etc. (but not code, image, etc.).


<div class="api-test-widget" data-widget-id="widget_put__chutes__chute_id_or_name_"></div>
<script type="application/json" data-widget-config="widget_put__chutes__chute_id_or_name_">{"endpoint":"/chutes/{chute_id_or_name}","method":"PUT","requiresAuth":true,"parameters":[{"name":"chute_id_or_name","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"tagline":{"anyOf":[{"type":"string","maxLength":1024},{"type":"null"}],"title":"Tagline","default":""},"readme":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Readme","default":""},"tool_description":{"anyOf":[{"type":"string","maxLength":16384},{"type":"null"}],"title":"Tool Description","default":""},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"}},"required":[]}}</script>

**Endpoint:** `PUT /chutes/{chute_id_or_name}`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id_or_name | string | Yes |  |
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tagline | string \| null | No |  |
| readme | string \| null | No |  |
| tool_description | string \| null | No |  |
| logo_id | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Share Chute

Share a chute with another user.


<div class="api-test-widget" data-widget-id="widget_post__chutes__chute_id__share"></div>
<script type="application/json" data-widget-config="widget_post__chutes__chute_id__share">{"endpoint":"/chutes/{chute_id}/share","method":"POST","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /chutes/{chute_id}/share`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| chute_id | string | Yes |  |
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

## Easy Deploy Vllm Chute

Easy/templated vLLM deployment.


<div class="api-test-widget" data-widget-id="widget_post__chutes_vllm"></div>
<script type="application/json" data-widget-config="widget_post__chutes_vllm">{"endpoint":"/chutes/vllm","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"model":{"type":"string","title":"Model"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"tagline":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tagline","default":""},"tool_description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tool Description"},"readme":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Readme","default":""},"public":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Public","default":true},"node_selector":{"anyOf":[{"$ref":"#/components/schemas/NodeSelector"},{"type":"null"}]},"engine_args":{"anyOf":[{"$ref":"#/components/schemas/VLLMEngineArgs"},{"type":"null"}]},"revision":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Revision"},"concurrency":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Concurrency","default":8}},"required":["model"]}}</script>

**Endpoint:** `POST /chutes/vllm`

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
| model | string | Yes |  |
| logo_id | string \| null | No |  |
| tagline | string \| null | No |  |
| tool_description | string \| null | No |  |
| readme | string \| null | No |  |
| public | boolean \| null | No |  |
| node_selector | NodeSelector \| null | No |  |
| engine_args | VLLMEngineArgs \| null | No |  |
| revision | string \| null | No |  |
| concurrency | integer \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Easy Deploy Diffusion Chute

Easy/templated diffusion deployment.


<div class="api-test-widget" data-widget-id="widget_post__chutes_diffusion"></div>
<script type="application/json" data-widget-config="widget_post__chutes_diffusion">{"endpoint":"/chutes/diffusion","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"model":{"type":"string","title":"Model"},"name":{"type":"string","title":"Name"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"tagline":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tagline","default":""},"tool_description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tool Description"},"readme":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Readme","default":""},"public":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Public","default":true},"node_selector":{"anyOf":[{"$ref":"#/components/schemas/NodeSelector"},{"type":"null"}]},"concurrency":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Concurrency","default":1}},"required":["model","name"]}}</script>

**Endpoint:** `POST /chutes/diffusion`

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
| model | string | Yes |  |
| name | string | Yes |  |
| logo_id | string \| null | No |  |
| tagline | string \| null | No |  |
| tool_description | string \| null | No |  |
| readme | string \| null | No |  |
| public | boolean \| null | No |  |
| node_selector | NodeSelector \| null | No |  |
| concurrency | integer \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Easy Deploy Tei Chute

Easy/templated text-embeddings-inference deployment.


<div class="api-test-widget" data-widget-id="widget_post__chutes_tei"></div>
<script type="application/json" data-widget-config="widget_post__chutes_tei">{"endpoint":"/chutes/tei","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"model":{"type":"string","title":"Model"},"endpoints":{"items":{"type":"string","enum":["embed","predict","rerank"]},"type":"array","minItems":1,"title":"Endpoints","description":"List of supported endpoints for this chute"},"revision":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Revision"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"},"tagline":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tagline","default":""},"tool_description":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Tool Description"},"readme":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Readme","default":""},"public":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Public","default":true},"node_selector":{"anyOf":[{"$ref":"#/components/schemas/NodeSelector"},{"type":"null"}]}},"required":["model","endpoints"]}}</script>

**Endpoint:** `POST /chutes/tei`

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
| model | string | Yes |  |
| endpoints | string[] | Yes | List of supported endpoints for this chute |
| revision | string \| null | No |  |
| logo_id | string \| null | No |  |
| tagline | string \| null | No |  |
| tool_description | string \| null | No |  |
| readme | string \| null | No |  |
| public | boolean \| null | No |  |
| node_selector | NodeSelector \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Bounties

List available bounties, if any.


<div class="api-test-widget" data-widget-id="widget_get__bounties_"></div>
<script type="application/json" data-widget-config="widget_get__bounties_">{"endpoint":"/bounties/","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /bounties/`

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
