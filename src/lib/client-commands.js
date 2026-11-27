import { shellQuote } from "./command-synthesis.js";
import { substitute } from "./cluster-endpoints.js";

const POOLING_EXAMPLES = {
  embed: {
    endpoint: "/v1/embeddings",
    body: { input: "Hello" },
    backend: "openai-embeddings",
    dataset: "random",
  },
  rerank: {
    endpoint: "/v1/rerank",
    body: {
      query: "What is the capital of France?",
      documents: ["Paris is the capital of France.", "Rome is the capital of Italy."],
    },
    backend: "vllm-rerank",
    dataset: "random-rerank",
  },
};

function optionValue(argv, flag) {
  for (let i = argv.length - 1; i >= 0; i--) {
    if (argv[i] === flag) return argv[i + 1];
    if (argv[i].startsWith(`${flag}=`)) return argv[i].slice(flag.length + 1);
  }
  return undefined;
}

function poolerTask(argv) {
  const task = optionValue(argv, "--pooler-config.task");
  if (task) return task;

  const config = optionValue(argv, "--pooler-config");
  if (!config) return undefined;
  try {
    return JSON.parse(config)?.task;
  } catch {
    return undefined;
  }
}

function poolingExample(argv, modelId, conversion) {
  const task = poolerTask(argv);
  if (task) {
    return task === "embed" ? POOLING_EXAMPLES.embed
      : task === "rerank" || task === "score" ? POOLING_EXAMPLES.rerank : null;
  }
  if (conversion === "embed") return POOLING_EXAMPLES.embed;

  // The recipe's embedding category covers several pooling tasks, including
  // classification. Only offer examples when the task can be identified.
  const name = modelId.split("/").at(-1).toLowerCase();
  if (/(?:^|[._-])rerank(?:er)?(?:$|[._-])/.test(name)) return POOLING_EXAMPLES.rerank;
  if (conversion === "classify" || conversion === "reward") return null;
  if (/(?:^|[._-])embed(?:ding)?s?(?:$|[._-])/.test(name)) return POOLING_EXAMPLES.embed;
  return null;
}

export function resolveClientCommands({ argv = [], prefillArgv = [], tasks = [], modelId, host, port, endpoints, omniTask }) {
  const alias = optionValue(argv, "--served-model-name") ?? optionValue(prefillArgv, "--served-model-name");
  const requestModel = substitute(alias || modelId, endpoints);
  // The benchmark loads its tokenizer from the checkpoint and sends the alias in API requests.
  const modelArgs = [`--model ${shellQuote(substitute(modelId, endpoints))}`];
  if (alias) modelArgs.push(`--served-model-name ${shellQuote(requestModel)}`);
  const benchModelArgs = modelArgs.join(" \\\n  ");

  const runner = optionValue(argv, "--runner");
  const conversion = optionValue(argv, "--convert");
  const pooling = runner !== "generate" && (
    runner === "pooling" || tasks.includes("embedding")
    || ["embed", "classify", "reward"].includes(conversion)
  );

  if (pooling && !omniTask) {
    const example = poolingExample(argv, modelId, conversion);
    if (!example) return { verifyCmd: null, benchCmd: null };

    const { endpoint, body, backend, dataset } = example;
    const payload = JSON.stringify({ model: requestModel, ...body }, null, 2).replaceAll("'", "'\\''");
    return {
      verifyCmd: `curl http://${host}:${port}${endpoint} \\
  -H "Content-Type: application/json" \\
  -d '${payload}'`,
      benchCmd: `vllm bench serve \\
  ${benchModelArgs} \\
  --host ${host} \\
  --port ${port} \\
  --backend ${backend} \\
  --endpoint ${endpoint} \\
  --dataset-name ${dataset} \\
  --random-input-len 128 \\
  --num-prompts 100 \\
  --max-concurrency 32`,
    };
  }

  return {
    verifyCmd: omniTask?.example?.({ host, port, modelId: requestModel }) ?? `curl http://${host}:${port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": ${JSON.stringify(requestModel).replaceAll("'", "'\\''")},
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'`,
    benchCmd: omniTask?.benchmark || `vllm bench serve \\
  ${benchModelArgs} \\
  --host ${host} \\
  --port ${port} \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 100 \\
  --max-concurrency 32`,
  };
}
