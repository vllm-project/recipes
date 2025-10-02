# [vLLM Recipes](https://docs.vllm.ai/projects/recipes)

This repo intends to host community maintained common recipes to run vLLM answering the question:
**How do I run model X on hardware Y for task Z?**

## Guides

### DeepSeek <img src="https://avatars.githubusercontent.com/u/148330874?s=200&v=4" alt="DeepSeek" width="16" height="16" style="vertical-align:middle;">
- [DeepSeek-V3, DeepSeek-R1](DeepSeek/DeepSeek-V3.md)
- [DeepSeek-V3.1](DeepSeek/DeepSeek-V3_1.md)
- [DeepSeek-V3.2-Exp](DeepSeek/DeepSeek-V3_2-Exp.md)
- [DeepSeek-V3.2-Exp - Complete Tutorial (Jupyter Notebook)](https://nbviewer.org/github/vllm-project/recipes/blob/main/DeepSeek/DeepSeek_v3_2_vLLM_getting_started_guide.ipynb)

### Ernie <img src="https://avatars.githubusercontent.com/u/13245940?v=4" alt="Ernie" width="16" height="16" style="vertical-align:middle;">
- [Ernie4.5](Ernie/Ernie4.5.md)
- [Ernie4.5-VL](Ernie/Ernie4.5-VL.md)

### GLM <img src="https://raw.githubusercontent.com/zai-org/GLM-4.5/refs/heads/main/resources/logo.svg" alt="GLM" width="16" height="16" style="vertical-align:middle;">
- [GLM-4.5, GLM-4.5-Air](GLM/GLM-4.5.md)
- [GLM-4.5V](GLM/GLM-4.5V.md)

### InternVL <img src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4" alt="InternVL" width="64" height="16">
- [InternVL3.5](InternVL/InternVL3_5.md)

### InternLM <img src="https://avatars.githubusercontent.com/u/135356492?s=200&v=4" alt="InternLM" width="16" height="16" style="vertical-align:middle;">
- [Intern-S1](InternLM/Intern-S1.md)

### Llama
- [Llama3.3-70B](Llama/Llama3.3-70B.md)
- [Llama4-Scout](Llama/Llama4-Scout.md)

### OpenAI <img src="https://avatars.githubusercontent.com/u/14957082?v=4" alt="OpenAI" width="16" height="16" style="vertical-align:middle;">
- [gpt-oss](OpenAI/GPT-OSS.md)

### Qwen <img src="https://qwenlm.github.io/favicon.png" alt="Qwen" width="16" height="16" style="vertical-align:middle;">
- [Qwen2.5-VL](Qwen/Qwen2.5-VL.md)
- [Qwen3-Coder-480B-A35B](Qwen/Qwen3-Coder-480B-A35B.md)
- [Qwen3-Next](Qwen/Qwen3-Next.md)
- [Qwen3-VL](Qwen/Qwen3-VL.md)

### Seed <img src="https://avatars.githubusercontent.com/u/4158466?s=200&v=4" alt="Seed" width="16" height="16" style="vertical-align:middle;">
- [Seed-OSS-36B](Seed/Seed-OSS-36B.md)

### Moonshotai <img src="https://avatars.githubusercontent.com/u/129152888?v=4" alt="Moonshotai" width="16" height="16" style="vertical-align:middle;">
- [Kimi-K2](moonshotai/Kimi-K2.md)

## Contributing
Please feel free to contribute by adding a new recipe or improving an existing one, just send us a PR!

While the repo is designed to be directly viewable in GitHub (Markdown files as first citizen), you can build the docs as web pages locally.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run mkdocs serve
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/vllm-project/recipes/blob/main/LICENSE) file for details.
