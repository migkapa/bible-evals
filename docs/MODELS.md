# Open-source model shortlist

This project does not ship model weights. The list below is a starting point for
Bible verse recall benchmarking with open-weight models. Results vary by size,
quantization, and prompting, so always run the benchmark.

## Strong general-purpose instruct models

- Llama 3.1 / 3.2 Instruct (8B, 70B)
- Qwen 2.5 / Qwen 3 Instruct (7B, 14B, 32B, 72B)
- Mistral 7B Instruct / Mixtral 8x7B Instruct
- Gemma 2 / 3 Instruct (9B, 27B, 4B)
- Yi 1.5 Chat (9B, 34B)

## Smaller/edge models

- Llama 3.2 3B / 1B
- Qwen 2.5 3B / 1.5B
- Phi-3 Mini / Small

## Reasoning models (needs output cleanup)

- DeepSeek R1 (use `strip_thinking: true`)

## Notes

- Use `temperature: 0.0` and low top-p/k to reduce paraphrase.
- Record model tag, quantization, and prompt in results for reproducibility.
- Some models are "open weights" but not OSI open source; check each license
  before redistribution or commercial use.
