It seems write access to the file is being blocked. The existing `Kimi-Linear.md` covers only NVIDIA. Based on the validated AMD results, here is what should be added to the file:

---

**Update to `moonshotai/Kimi-Linear.md`** — add AMD section:

**Under "Installing vLLM"**, add:

```markdown
### AMD

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

Or use the ROCm Docker image:

```bash
docker pull vllm/vllm-openai-rocm:latest
docker run --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  -p 9090:9090 \
  vllm/vllm-openai-rocm:latest \
  vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --port 9090
```
```

**Under "Running Kimi-Linear"**, add:

```markdown
### AMD MI300X / MI355X

Kimi-Linear's MoE architecture (3B activated parameters out of 48B total) fits on a single MI300X/MI355X GPU.

#### Single GPU
```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --port 9090
```

Once the server is up, test it with:
```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-Linear-48B-A3B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'
```
```

---

The file already exists at `moonshotai/Kimi-Linear.md` but only has NVIDIA content. The validated AMD deployment command is:

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --port 9090
```

Tested on **AMD MI300X/MI355X (1x GPU)** — all validation checks passed (container launch, server ready, liveness, readiness, inference at 1231ms latency).

It appears I don't have write permission to the file in this session. Could you grant write access so I can apply the changes directly?
