# vLLM Recipes

This repo intends to host community maintained common recipes to run vLLM anwering the question:
**How do I run model Y on hardware Y for task Z?**

## Guides

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