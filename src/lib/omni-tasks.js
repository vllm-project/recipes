/**
 * Built-in catalog of vllm-omni online-serving task templates.
 *
 * Each task id maps to:
 *   - label:       short pill text ("Text → Image")
 *   - endpoint:    HTTP path the model handler binds to ("/v1/images/generations")
 *   - method:      always "POST" today; carried so future GET endpoints don't need a schema bump
 *   - example:     ({ host, port, modelId, prompt? }) → ready-to-paste curl command string
 *
 * Recipes opt in by listing task ids under `omni.tasks` in their YAML; per-task
 * overrides (model_id swap, extra args, custom curl body) live next to the id.
 * Source: docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/
 */

const SAMPLE_PROMPTS = {
  t2i: "a dragon flying over the Green Mountains at sunset",
  i2i: "Convert this image to a watercolor painting with soft pastel tones",
  t2v: "A cinematic view of a futuristic city at sunset, smooth camera pan",
  i2v: "A bear playing with yarn, smooth motion",
  ti2v: "A bear playing with yarn, smooth motion",
  t2a: "The sound of a cat purring softly",
};

function jsonCurl({ host, port, endpoint, body }) {
  // jq-friendly pretty-printed JSON so the rendered curl reads cleanly even
  // when copy-pasted into a terminal.
  const indented = JSON.stringify(body, null, 2)
    .split("\n")
    .map((line, i) => (i === 0 ? line : `    ${line}`))
    .join("\n");
  return `curl -X POST http://${host}:${port}${endpoint} \\
  -H "Content-Type: application/json" \\
  -d '${indented}'`;
}

function formCurl({ host, port, endpoint, fields, outFile }) {
  const lines = [`curl -X POST http://${host}:${port}${endpoint}`];
  for (const [k, v] of Object.entries(fields)) {
    lines.push(`  -F "${k}=${v}"`);
  }
  if (outFile) lines.push(`  --output ${outFile}`);
  return lines.join(" \\\n");
}

export const OMNI_TASKS = {
  t2i: {
    label: "Text → Image",
    endpoint: "/v1/images/generations",
    method: "POST",
    example: ({ host, port, prompt }) =>
      `${jsonCurl({
        host,
        port,
        endpoint: "/v1/images/generations",
        body: {
          prompt: prompt || SAMPLE_PROMPTS.t2i,
          size: "1024x1024",
          seed: 42,
        },
      })} \\
  | jq -r '.data[0].b64_json' | base64 -d > output.png`,
  },

  i2i: {
    label: "Image → Image",
    endpoint: "/v1/chat/completions",
    method: "POST",
    example: ({ host, port, modelId, prompt }) =>
      jsonCurl({
        host,
        port,
        endpoint: "/v1/chat/completions",
        body: {
          model: modelId,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: prompt || SAMPLE_PROMPTS.i2i },
                {
                  type: "image_url",
                  image_url: {
                    url: "data:image/png;base64,<BASE64_IMAGE>",
                  },
                },
              ],
            },
          ],
          extra_body: {
            height: 1024,
            width: 1024,
            num_inference_steps: 50,
            guidance_scale: 4.5,
            seed: 42,
          },
        },
      }),
  },

  t2v: {
    label: "Text → Video",
    endpoint: "/v1/videos",
    method: "POST",
    example: ({ host, port, prompt }) =>
      formCurl({
        host,
        port,
        endpoint: "/v1/videos/sync",
        fields: {
          prompt: prompt || SAMPLE_PROMPTS.t2v,
          width: 832,
          height: 480,
          num_frames: 81,
          fps: 16,
          num_inference_steps: 40,
          guidance_scale: 4.0,
          seed: 42,
        },
        outFile: "output.mp4",
      }),
  },

  i2v: {
    label: "Image → Video",
    endpoint: "/v1/videos",
    method: "POST",
    example: ({ host, port, prompt }) =>
      formCurl({
        host,
        port,
        endpoint: "/v1/videos/sync",
        fields: {
          prompt: prompt || SAMPLE_PROMPTS.i2v,
          input_reference: "@/path/to/input.png",
          width: 832,
          height: 480,
          num_frames: 81,
          fps: 16,
          num_inference_steps: 40,
          guidance_scale: 5.0,
          seed: 42,
        },
        outFile: "output.mp4",
      }),
  },

  ti2v: {
    label: "Text+Image → Video",
    endpoint: "/v1/videos",
    method: "POST",
    example: ({ host, port, prompt }) =>
      formCurl({
        host,
        port,
        endpoint: "/v1/videos/sync",
        fields: {
          prompt: prompt || SAMPLE_PROMPTS.ti2v,
          input_reference: "@/path/to/input.png",
          width: 832,
          height: 480,
          num_frames: 81,
          fps: 16,
          num_inference_steps: 40,
          guidance_scale: 5.0,
          seed: 42,
        },
        outFile: "output.mp4",
      }),
  },

  t2a: {
    label: "Text → Audio",
    endpoint: "/v1/audio/generate",
    method: "POST",
    example: ({ host, port, prompt }) =>
      `${jsonCurl({
        host,
        port,
        endpoint: "/v1/audio/generate",
        body: {
          input: prompt || SAMPLE_PROMPTS.t2a,
          audio_length: 10.0,
          negative_prompt: "Low quality",
          guidance_scale: 7.0,
          num_inference_steps: 100,
          seed: 42,
        },
      })} --output output.wav`,
  },
};

/**
 * Resolve a recipe's task list to the catalog entries it actually uses.
 *
 * Accepts either:
 *   omni: { tasks: ["t2i"] }                              — bare ids
 *   omni: { tasks: [{ id: "i2i", model_id, vram_minimum_gb, description,
 *                     extra_args, curl }, ...] }           — per-task overrides
 *
 * Per-task fields:
 *   - model_id          swap the served checkpoint (Wan2.2 picks a different
 *                       HF repo for T2V/I2V/TI2V)
 *   - vram_minimum_gb   drives the hardware-fit hint when set (otherwise
 *                       falls back to the recipe's default variant VRAM)
 *   - description       short blurb shown next to the task pill
 *   - extra_args        appended to the rendered `vllm serve --omni` command
 *   - curl              static curl override (otherwise the built-in renderer
 *                       interpolates host/port/modelId into a sample request)
 *
 * Returns: [{ id, label, endpoint, method, modelId?, vramMinimumGb?,
 *             description?, extraArgs?, example }]
 */
export function resolveOmniTasks(recipe) {
  const decl = recipe?.omni?.tasks;
  if (!Array.isArray(decl) || decl.length === 0) return [];
  const out = [];
  for (const t of decl) {
    const id = typeof t === "string" ? t : t?.id;
    if (!id) continue;
    const base = OMNI_TASKS[id];
    if (!base) continue;
    const override = typeof t === "string" ? {} : t;
    const customCurl = override.curl;
    out.push({
      id,
      label: override.label || base.label,
      endpoint: override.endpoint || base.endpoint,
      method: base.method,
      modelId: override.model_id,
      vramMinimumGb: override.vram_minimum_gb,
      description: override.description,
      extraArgs: override.extra_args || [],
      example: customCurl ? () => customCurl : base.example,
    });
  }
  return out;
}

/**
 * Returns true when the recipe opts into vllm-omni online serving.
 * Distinct from `meta.tasks` containing "omni" — a recipe can be omni-tagged
 * (offline-only doc) without yet declaring `omni.tasks` for the command builder.
 */
export function hasOmniTasks(recipe) {
  return resolveOmniTasks(recipe).length > 0;
}

/**
 * Default render-time port for the omni command card.
 * Same 8000 the rest of the site uses; recipe authors can override via
 * `omni.port` (rendered into the served --port flag).
 */
export const OMNI_DEFAULT_PORT = 8000;
