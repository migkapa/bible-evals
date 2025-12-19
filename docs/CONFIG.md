# Configuration (`config.yaml`)

## `data`

- `taxonomy_path`: book list + aliases used to compute canonical IDs.
- `versions.<key>.raw_path`: path to the verse JSON file for that translation.
- `versions.<key>.name`: display name used in prompts.

## `models[]`

Each model entry selects a connector and options.

### Ollama

```yaml
models:
  - name: "ollama:llama3"
    connector: "ollama"
    options:
      base_url: "http://localhost:11434"
      model: "llama3"
      temperature: 0.0
```

### OpenAI-compatible (vLLM, llama.cpp server, etc.)

```yaml
models:
  - name: "vllm:llama-3"
    connector: "openai-compatible"
    options:
      base_url: "http://localhost:8000"
      model: "meta-llama/Meta-Llama-3-8B-Instruct"
      temperature: 0.0
```

## `prompts`

Defines templates for `naive`, `constraint`, and `system2` regimes.

## `eval`

- `version`: selects `data.versions.<key>`.
- `prompt`: one of `naive`, `constraint`, `system2`.
- `sample.count`: number of verses to evaluate.
- `sample.seed`: RNG seed.
- `sample.stratified`: when true, balances OT/NT (book index ≤39 vs ≥40).

## `engine`

- `max_retries`: retry count for transient connector failures.
- `backoff_s`: exponential backoff base delay.

