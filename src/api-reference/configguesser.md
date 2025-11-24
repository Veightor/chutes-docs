# Configguesser API Reference

This section covers all endpoints related to configguesser.


## Analyze Model

Attempt to guess required GPU count and VRAM for a model on huggingface, assuming safetensors format.


<div class="api-test-widget" data-widget-id="widget_get__guess_vllm_config"></div>
<script type="application/json" data-widget-config="widget_get__guess_vllm_config">{"endpoint":"/guess/vllm_config","method":"GET","parameters":[{"name":"model","type":"string","required":true,"description":"","in":"query"}],"requestBody":null}</script>

**Endpoint:** `GET /guess/vllm_config`

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes |  |


### Responses

| Status Code | Description |
|-------------|-------------|
| 200 | Successful Response |
| 422 | Validation Error |

---
