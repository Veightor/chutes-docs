# Users API Reference

This section covers all endpoints related to users.


## Get User Growth


<div class="api-test-widget" data-widget-id="widget_get__users_growth"></div>
<script type="application/json" data-widget-config="widget_get__users_growth">{"endpoint":"/users/growth","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /users/growth`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Admin User Id Lookup


<div class="api-test-widget" data-widget-id="widget_get__users_user_id_lookup"></div>
<script type="application/json" data-widget-config="widget_get__users_user_id_lookup">{"endpoint":"/users/user_id_lookup","method":"GET","requiresAuth":true,"parameters":[{"name":"username","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/user_id_lookup`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| username | string | Yes |  |
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

## Admin Balance Change


<div class="api-test-widget" data-widget-id="widget_post__users_admin_balance_change"></div>
<script type="application/json" data-widget-config="widget_post__users_admin_balance_change">{"endpoint":"/users/admin_balance_change","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"user_id":{"type":"string","title":"User Id"},"amount":{"type":"number","title":"Amount"},"reason":{"type":"string","title":"Reason"}},"required":["user_id","amount","reason"]}}</script>

**Endpoint:** `POST /users/admin_balance_change`

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
| user_id | string | Yes |  |
| amount | number | Yes |  |
| reason | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Admin Quotas Change


<div class="api-test-widget" data-widget-id="widget_post__users__user_id__quotas"></div>
<script type="application/json" data-widget-config="widget_post__users__user_id__quotas">{"endpoint":"/users/{user_id}/quotas","method":"POST","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/{user_id}/quotas`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin Discounts Change


<div class="api-test-widget" data-widget-id="widget_post__users__user_id__discounts"></div>
<script type="application/json" data-widget-config="widget_post__users__user_id__discounts">{"endpoint":"/users/{user_id}/discounts","method":"POST","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/{user_id}/discounts`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Admin Enable Invoicing


<div class="api-test-widget" data-widget-id="widget_post__users__user_id__enable_invoicing"></div>
<script type="application/json" data-widget-config="widget_post__users__user_id__enable_invoicing">{"endpoint":"/users/{user_id}/enable_invoicing","method":"POST","requiresAuth":true,"parameters":[{"name":"user_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/{user_id}/enable_invoicing`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| user_id | string | Yes |  |
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

## Me

Get a detailed response for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_me"></div>
<script type="application/json" data-widget-config="widget_get__users_me">{"endpoint":"/users/me","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me`

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

## Delete My User

Delete account.


<div class="api-test-widget" data-widget-id="widget_delete__users_me"></div>
<script type="application/json" data-widget-config="widget_delete__users_me">{"endpoint":"/users/me","method":"DELETE","parameters":[{"name":"Authorization","type":"string","required":true,"description":"Authorization header","in":"header"}],"requestBody":null}</script>

**Endpoint:** `DELETE /users/me`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Authorization | string | Yes | Authorization header |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## My Quotas

Load quotas for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_me_quotas"></div>
<script type="application/json" data-widget-config="widget_get__users_me_quotas">{"endpoint":"/users/me/quotas","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/quotas`

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

## My Discounts

Load discounts for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_me_discounts"></div>
<script type="application/json" data-widget-config="widget_get__users_me_discounts">{"endpoint":"/users/me/discounts","method":"GET","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/discounts`

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

## Chute Quota Usage

Check the current quota usage for a chute.


<div class="api-test-widget" data-widget-id="widget_get__users_me_quota_usage__chute_id_"></div>
<script type="application/json" data-widget-config="widget_get__users_me_quota_usage__chute_id_">{"endpoint":"/users/me/quota_usage/{chute_id}","method":"GET","requiresAuth":true,"parameters":[{"name":"chute_id","type":"string","required":true,"description":"","in":"path"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/quota_usage/{chute_id}`

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

## Set Logo

Get a detailed response for the current user.


<div class="api-test-widget" data-widget-id="widget_get__users_set_logo"></div>
<script type="application/json" data-widget-config="widget_get__users_set_logo">{"endpoint":"/users/set_logo","method":"GET","requiresAuth":true,"parameters":[{"name":"logo_id","type":"string","required":true,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/set_logo`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| logo_id | string | Yes |  |
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

## Check Username

Check if a username is valid and available.


<div class="api-test-widget" data-widget-id="widget_get__users_name_check"></div>
<script type="application/json" data-widget-config="widget_get__users_name_check">{"endpoint":"/users/name_check","method":"GET","parameters":[{"name":"username","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /users/name_check`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| username | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Register

Register a user.


<div class="api-test-widget" data-widget-id="widget_post__users_register"></div>
<script type="application/json" data-widget-config="widget_post__users_register">{"endpoint":"/users/register","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string","required":true,"description":"The hotkey of the user","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"username":{"type":"string","title":"Username"},"coldkey":{"type":"string","title":"Coldkey"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"}},"required":["username","coldkey"]}}</script>

**Endpoint:** `POST /users/register`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string | Yes | The hotkey of the user |
| X-Chutes-Signature | string \| null | No |  |
| X-Chutes-Nonce | string \| null | No |  |
| Authorization | string \| null | No |  |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| username | string | Yes |  |
| coldkey | string | Yes |  |
| logo_id | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Admin Create User

Create a new user manually from an admin account, no bittensor stuff necessary.


<div class="api-test-widget" data-widget-id="widget_post__users_create_user"></div>
<script type="application/json" data-widget-config="widget_post__users_create_user">{"endpoint":"/users/create_user","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"username":{"type":"string","title":"Username"},"logo_id":{"anyOf":[{"type":"string"},{"type":"null"}],"title":"Logo Id"}},"required":["username"]}}</script>

**Endpoint:** `POST /users/create_user`

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
| username | string | Yes |  |
| logo_id | string \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## Change Fingerprint

Reset a user's fingerprint using either the hotkey or coldkey.


<div class="api-test-widget" data-widget-id="widget_post__users_change_fingerprint"></div>
<script type="application/json" data-widget-config="widget_post__users_change_fingerprint">{"endpoint":"/users/change_fingerprint","method":"POST","parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Coldkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string","required":true,"description":"Nonce","in":"header"},{"name":"X-Chutes-Signature","type":"string","required":true,"description":"Hotkey signature","in":"header"}],"requestBody":{"type":"object","properties":{"fingerprint":{"type":"string","title":"Fingerprint"}},"required":["fingerprint"]}}</script>

**Endpoint:** `POST /users/change_fingerprint`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| X-Chutes-Hotkey | string \| null | No |  |
| X-Chutes-Coldkey | string \| null | No |  |
| X-Chutes-Nonce | string | Yes | Nonce |
| X-Chutes-Signature | string | Yes | Hotkey signature |


### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| fingerprint | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Fingerprint Login

Exchange the fingerprint for a JWT.


<div class="api-test-widget" data-widget-id="widget_post__users_login"></div>
<script type="application/json" data-widget-config="widget_post__users_login">{"endpoint":"/users/login","method":"POST","parameters":[],"requestBody":null}</script>

**Endpoint:** `POST /users/login`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Change Bt Auth

Change the bittensor hotkey/coldkey associated with an account via fingerprint auth.


<div class="api-test-widget" data-widget-id="widget_post__users_change_bt_auth"></div>
<script type="application/json" data-widget-config="widget_post__users_change_bt_auth">{"endpoint":"/users/change_bt_auth","method":"POST","parameters":[{"name":"Authorization","type":"string","required":true,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `POST /users/change_bt_auth`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| Authorization | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---

## Update Squad Access

Enable squad access.


<div class="api-test-widget" data-widget-id="widget_put__users_squad_access"></div>
<script type="application/json" data-widget-config="widget_put__users_squad_access">{"endpoint":"/users/squad_access","method":"PUT","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `PUT /users/squad_access`

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

## List Usage

List usage summary data.


<div class="api-test-widget" data-widget-id="widget_get__users_me_usage"></div>
<script type="application/json" data-widget-config="widget_get__users_me_usage">{"endpoint":"/users/me/usage","method":"GET","requiresAuth":true,"parameters":[{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"per_chute","type":"boolean \\| null","required":false,"description":"","in":"query"},{"name":"chute_id","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"start_date","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"end_date","type":"string \\| null","required":false,"description":"","in":"query"},{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":null}</script>

**Endpoint:** `GET /users/me/usage`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer \| null | No |  |
| limit | integer \| null | No |  |
| per_chute | boolean \| null | No |  |
| chute_id | string \| null | No |  |
| start_date | string \| null | No |  |
| end_date | string \| null | No |  |
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
