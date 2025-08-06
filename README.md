# [vLLM Recipes](https://docs.vllm.ai/projects/recipes)

This repo intends to host community maintained common recipes to run vLLM answering the question:
**How do I run model Y on hardware Y for task Z?**

## Guides

### OpenAI <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/OpenAI_logo_2025_%28symbol%29.svg/2585px-OpenAI_logo_2025_%28symbol%29.svg.png" alt="OpenAI" width="16" height="16" style="vertical-align:middle;"> 
- [gpt-oss](OpenAI/GPT-OSS.md)

### DeepSeek <img src="https://avatars.githubusercontent.com/u/148330874?s=200&v=4" alt="DeepSeek" width="16" height="16" style="vertical-align:middle;">
- [DeepSeek-V3, DeepSeek-R1](DeepSeek/DeepSeek-V3.md)

### Llama
- [Llama3.3-70B](Llama/Llama3.3-70B.md)
- [Llama4-Scout](Llama/Llama4-Scout.md)

### Qwen <img src="https://qwenlm.github.io/favicon.png" alt="Qwen" width="16" height="16" style="vertical-align:middle;">
- [Qwen3-Coder-480B-A35B](Qwen/Qwen3-Coder-480B-A35B.md)

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
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.