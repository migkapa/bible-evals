# Connectors

## Ollama

Uses `POST /api/chat` on the configured `base_url` (default `http://localhost:11434`).

Recommended model settings for verbatim recall benchmarks:

- `temperature: 0.0`
- low top-p / top-k (if your backend supports it)
- for models that emit `<think>...</think>` in `content`, enable `strip_thinking: true` to mimic `ollama run --hidethinking` for scoring.

Model shortlist: see `docs/MODELS.md` for open-weight model families commonly
used in verse-recall benchmarking.

## OpenAI-compatible servers

Targets `POST /v1/chat/completions`, which is supported by many local servers:

- vLLM (`--api-key` optional)
- llama.cpp server
- text-generation-inference (OpenAI adapter)

Set `OPENAI_API_KEY` if your server requires auth (configurable via `api_key_env`).

## Reference (offline baseline)

Connector: `reference`

Useful for smoke-testing the pipeline without a running model server.

```yaml
models:
  - name: "reference:verbatim"
    connector: "reference"
    options:
      mode: "verbatim"   # verbatim | chatter | empty
```

## Thinking models (DeepSeek R1)

Some models emit reasoning directly inside `message.content` using `<think>...</think>` tags.
For fair verse-text evaluation, you usually want to score the **final answer** while still retaining the raw output.

In `config.yaml`, set:

```yaml
options:
  strip_thinking: true
```

The evaluator will:

- save `prediction_raw` (with thinking)
- save `prediction` (scored, with thinking stripped)
- report `clean_output_rate` so you can see whether the model produced clean output without postprocessing
