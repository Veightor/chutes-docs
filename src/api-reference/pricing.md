# Pricing API Reference

This section covers all endpoints related to pricing.


## Get Quota Unlock Amount

Amount, in USD, of payment history that provides the "free" tier quota.


<div class="api-test-widget" data-widget-id="widget_get__quota_unlock_amount"></div>
<script type="application/json" data-widget-config="widget_get__quota_unlock_amount">{"endpoint":"/quota_unlock_amount","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /quota_unlock_amount`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Fmv

Get the current FMV for tao.


<div class="api-test-widget" data-widget-id="widget_get__fmv"></div>
<script type="application/json" data-widget-config="widget_get__fmv">{"endpoint":"/fmv","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /fmv`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Pricing

Get the current compute unit pricing.


<div class="api-test-widget" data-widget-id="widget_get__pricing"></div>
<script type="application/json" data-widget-config="widget_get__pricing">{"endpoint":"/pricing","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /pricing`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Get Developer Deposit

Get the USD/tao amount required to enable developer mode.


<div class="api-test-widget" data-widget-id="widget_get__developer_deposit"></div>
<script type="application/json" data-widget-config="widget_get__developer_deposit">{"endpoint":"/developer_deposit","method":"GET","parameters":[],"requestBody":null}</script>

**Endpoint:** `GET /developer_deposit`


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |

---

## Return Developer Deposit


<div class="api-test-widget" data-widget-id="widget_post__return_developer_deposit"></div>
<script type="application/json" data-widget-config="widget_post__return_developer_deposit">{"endpoint":"/return_developer_deposit","method":"POST","requiresAuth":true,"parameters":[{"name":"X-Chutes-Hotkey","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Signature","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"X-Chutes-Nonce","type":"string \\| null","required":false,"description":"","in":"header"},{"name":"Authorization","type":"string \\| null","required":false,"description":"","in":"header"}],"requestBody":{"type":"object","properties":{"address":{"type":"string","title":"Address"}},"required":["address"]}}</script>

**Endpoint:** `POST /return_developer_deposit`

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
| address | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

### Authentication

This endpoint requires authentication.

---

## List Payments

List all payments.


<div class="api-test-widget" data-widget-id="widget_get__payments"></div>
<script type="application/json" data-widget-config="widget_get__payments">{"endpoint":"/payments","method":"GET","parameters":[{"name":"page","type":"integer \\| null","required":false,"description":"","in":"query"},{"name":"limit","type":"integer \\| null","required":false,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /payments`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | integer \| null | No |  |
| limit | integer \| null | No |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
