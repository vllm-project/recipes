// JSON Schema (draft-07) for a recipe YAML. Generated from the taxonomy and
// strategy YAMLs so hardware / strategy / frontend / task enums never drift
// from the renderer, published at /recipe.schema.json for editors, and
// enforced on every recipe by scripts/build-recipes-api.mjs.
//
// It checks structure — known keys (a misspelled key is an error, since the
// renderer silently ignores unknown ones), value types, and vocabulary ids.
// Cross-field rules (mode references, hardware semantics, frontends) stay in
// the build script's validators. SCHEMA.md documents every field.

const strings = { type: "array", items: { type: "string" } };
const envValue = { type: ["string", "number", "boolean"] };
const envMap = {
  type: "object",
  propertyNames: { pattern: "^[A-Za-z_][A-Za-z0-9_]*$" },
  additionalProperties: envValue,
};
const date = { type: "string", pattern: "^\\d{4}-\\d{2}-\\d{2}$" };

function obj(properties, { required, extra = false, description } = {}) {
  return {
    type: "object",
    ...(description ? { description } : {}),
    properties,
    ...(required ? { required } : {}),
    additionalProperties: extra,
  };
}

// Map whose keys are drawn from `keys` (a property-name schema).
function keyed(keys, value, description) {
  return { type: "object", ...(description ? { description } : {}), propertyNames: keys, additionalProperties: value };
}

export function buildRecipeSchema({ taxonomy, strategies }) {
  const profiles = taxonomy?.hardware_profiles || {};
  const profileIds = Object.keys(profiles);
  const generations = [...new Set(Object.values(profiles).map((p) => p.generation).filter(Boolean))];
  const brands = [...new Set(Object.values(profiles).map((p) => p.brand).filter(Boolean))];
  const archs = [...new Set(Object.values(profiles).map((p) => p.compute_arch).filter(Boolean))];
  const frontends = Object.keys(taxonomy?.frontends || {});
  const tasks = (taxonomy?.tasks || []).map((t) => (typeof t === "string" ? t : t.id));
  const kvOffload = Object.keys(taxonomy?.kv_offload || {});
  const all = Object.values(strategies || {});
  const kvStores = all.filter((s) => s.deploy_type === "kv_store_lb").map((s) => s.name);
  const serving = all.filter((s) => s.deploy_type !== "kv_store_lb").map((s) => s.name);

  const gpuId = { enum: profileIds };
  // Hardware-keyed maps accept an exact profile id, a generation, a lowercase
  // brand, `default`, or an `arch:<compute_arch>` selector.
  const hwKey = {
    anyOf: [
      { enum: [...new Set([...profileIds, ...generations, ...brands.map((b) => b.toLowerCase()), "default"])] },
      { pattern: `^arch:(${archs.join("|")})$` },
    ],
  };
  const strategyId = { enum: serving };
  const when = obj({
    hardware: { type: "array", minItems: 1, items: { anyOf: [{ enum: [...new Set([...profileIds, ...generations, ...brands.map((b) => b.toLowerCase())])] }, { pattern: `^arch:(${archs.join("|")})$` }] } },
    strategies: { type: "array", minItems: 1, items: strategyId },
    variants: { type: "array", minItems: 1, items: { type: "string" } },
    features: { type: "array", minItems: 1, items: { type: "string" } },
  });
  const status = { enum: ["verified", "supported", "unsupported"] };
  const dcpSizes = { type: "array", items: { type: "integer", minimum: 2 }, uniqueItems: true };
  const intOrHwMap = { anyOf: [{ type: "integer", minimum: 1 }, keyed(hwKey, { type: "integer", minimum: 1 })] };
  const argsEnv = obj({ args: strings, env: envMap });
  const extraArgsEnv = (more = {}) => obj({ extra_args: strings, extra_env: envMap, ...more });

  const mode = obj({
    label: { type: "string" },
    description: { type: "string" },
    args: strings,
    env: envMap,
    variants: strings,
    hardware: keyed(hwKey, { enum: ["unsupported", "supported"] }),
    hardware_overrides: keyed(hwKey, argsEnv),
  });

  const feature = obj({
    description: { type: "string" },
    label: { type: "string" },
    args: strings,
    env: envMap,
    strategies: { type: "array", items: strategyId },
    hardware: keyed(hwKey, { enum: ["unsupported", "supported"] }),
    hardware_overrides: keyed(hwKey, argsEnv),
    default_mode: { type: "string" },
    modes: { type: "object", additionalProperties: mode },
    companion: obj({ label: { type: "string" }, description: { type: "string" }, command: { type: "string" } }, { required: ["label", "command"] }),
  }, { required: ["description"] });

  const variant = obj({
    precision: { type: "string", description: "bf16 | fp16 | fp8 | int8 | awq | gptq | int4 | nvfp4 | fp4 | mxfp4 | mxfp8 | …" },
    vram_minimum_gb: { type: "number", minimum: 0 },
    description: { type: "string" },
    label: { type: "string" },
    model_id: { type: "string", pattern: "^[^/\\s]+/[^/\\s]+$" },
    extra_args: strings,
    extra_env: envMap,
    docker_image: { type: "string" },
    min_vllm_version: { type: "string" },
    nightly_required: { type: "boolean" },
    tp: intOrHwMap,
    supported_hardware: { type: "array", items: { anyOf: [gpuId, { pattern: `^arch:(${archs.join("|")})$` }] } },
    precision_hardware: obj({ brand: { enum: brands }, generation: { enum: generations } }),
    default_modes: { type: "object", additionalProperties: { type: "string" } },
    // Generation keys carry extra_args/extra_env; exact GPU ids may also pin an
    // image, a GPU-resident VRAM budget, or hide the pip install.
    hardware_overrides: keyed(hwKey, extraArgsEnv({
      vram_minimum_gb: { type: "number", minimum: 0 },
      docker_image: { type: "string" },
      docker_serve_command: { type: "string" },
      install: obj({ pip: { type: "boolean" }, docker: { type: "boolean" } }),
    })),
  }, { required: ["precision", "vram_minimum_gb"] });

  const installTab = { anyOf: [{ type: "boolean" }, { type: "null" }, obj({ command: { type: "string" }, note: { type: "string" } })] };

  const pdMode = { enum: ["tp", "tep", "dep"] };
  const pdRole = obj({
    nodes: intOrHwMap,
    parallelism: { anyOf: [pdMode, keyed(hwKey, pdMode)] },
    tp: { type: "integer", minimum: 1 },
    vllm_args: strings,
    env: envMap,
    parallel_flag: { type: "string" },
    hardware_overrides: keyed(hwKey, extraArgsEnv()),
  });

  const strategyOverride = obj({
    tp: intOrHwMap,
    extra_args: strings,
    extra_env: envMap,
    env: envMap,
    vllm_args: strings,
    hardware_overrides: keyed(hwKey, extraArgsEnv()),
    prefill: pdRole,
    decode: pdRole,
    router: obj({ intra_node_data_parallel_size: { type: "integer", minimum: 1 } }),
    pool_modes: { type: "array", items: { enum: ["tp", "tep", "dep"] } },
  });

  const omniTask = {
    anyOf: [
      { type: "string" },
      obj({
        id: { type: "string" },
        key: { type: "string" },
        label: { type: "string" },
        model_id: { type: "string" },
        vram_minimum_gb: { type: "number", minimum: 0 },
        description: { type: "string" },
        extra_args: strings,
        curl: { type: "string" },
        benchmark: { type: "string" },
        endpoint: { type: "string" },
        axes: { type: "object", additionalProperties: { type: "string" } },
        // Resolved by exact profile id only (resolveOmniTaskForHardware).
        hardware_overrides: keyed(gpuId, obj({ curl: { type: "string" }, extra_args: strings, vram_minimum_gb: { type: "number", minimum: 0 } })),
      }, { required: ["id"] }),
    ],
  };

  return {
    $schema: "http://json-schema.org/draft-07/schema#",
    $id: "https://recipes.vllm.ai/recipe.schema.json",
    title: "vLLM recipe",
    description: "A recipe YAML at models/<hf_org>/<hf_repo>.yaml. Field reference: https://github.com/vllm-project/recipes/blob/main/SCHEMA.md",
    type: "object",
    required: ["meta", "model", "variants", "compatible_strategies", "guide"],
    additionalProperties: false,
    properties: {
      meta: obj({
        title: { type: "string" },
        slug: { type: "string" },
        provider: { type: "string" },
        description: { type: "string" },
        date_added: date,
        date_updated: date,
        difficulty: { type: "string", description: "Deprecated: nothing reads it. Omit it in new recipes." },
        tasks: { type: "array", minItems: 1, items: { enum: tasks } },
        performance_headline: { type: "string" },
        related_recipes: strings,
        default_hardware: gpuId,
        default_variant: keyed(hwKey, { type: "string" }),
        hardware: keyed(gpuId, status, "verified = tested end-to-end; supported = declared (lists a restricted profile) without a verified claim; unsupported = confirmed blocker today"),
        platforms: { type: "array", items: { anyOf: [{ type: "string" }, obj({ id: { type: "string" }, install: { type: "string" }, url: { type: "string" }, blurb: { type: "string" }, script: { type: "string" } }, { required: ["id"] })] } },
      }, { required: ["title", "provider", "description", "date_updated", "tasks"] }),

      model: {
        ...obj({
          model_id: { type: "string", pattern: "^[^/\\s]+/[^/\\s]+$" },
          min_vllm_version: { type: "string" },
          architecture: { enum: ["dense", "moe"] },
          parameter_count: { type: "string" },
          active_parameters: { type: "string" },
          context_length: { type: "integer", minimum: 0 },      // 0 for omni (non-LLM) checkpoints
          base_args: strings,
          base_env: envMap,
          default_frontend: { enum: frontends },
          docker_image: { anyOf: [{ type: "string" }, { type: "object", additionalProperties: { type: "string" } }] },
          docker_source: obj({ mount_path: { type: "string" } }),
          install: obj({ pip: installTab, docker: installTab }),
          nightly_required: { type: "boolean" },
          flashinfer_autotune: { type: "boolean" },
          smg_grpc: obj({ reasoning_parser: { type: "string" }, tool_call_parser: { type: "string" } }),
          supports_dcp: { type: "boolean" },
        }, { required: ["model_id", "architecture", "parameter_count"] }),
        // `<frontend>_frontend: unsupported` opts a non-default frontend out.
        patternProperties: { [`^(${frontends.join("|")})_frontend$`]: { const: "unsupported" } },
      },

      dependencies: {
        type: "array",
        items: obj({
          command: { type: "string" },
          note: { type: "string" },
          optional: { type: "boolean" },
          brand: { anyOf: [{ enum: brands }, { type: "array", items: { enum: brands } }] },
          install_modes: { type: "array", items: { enum: ["pip", "docker"] } },
        }, { required: ["command"] }),
      },

      features: { type: "object", additionalProperties: feature },
      opt_in_features: strings,
      hardware_opt_in_features: keyed(gpuId, strings),         // looked up by exact profile id
      variants: { type: "object", required: ["default"], additionalProperties: variant },
      compatible_strategies: { type: "array", items: strategyId },
      decode_context_parallel: obj({
        sizes: { anyOf: [dcpSizes, keyed(hwKey, dcpSizes)] },
        comm_backend: { type: "string" },
        match_tp: { type: "boolean" },
      }, { required: ["sizes"] }),
      default_strategy: strategyId,
      strategy_min_gpus: {
        anyOf: [
          { type: "integer", minimum: 1 },
          keyed({ anyOf: [strategyId, gpuId] }, { anyOf: [{ type: "integer", minimum: 1 }, keyed(strategyId, { type: "integer", minimum: 1 })] }),
        ],
      },
      kv_cache_strategy_hardware: keyed({ enum: kvStores }, keyed(gpuId, { const: "unsupported" })),
      kv_offload_support: keyed({ enum: kvOffload }, status),
      strategy_hardware: keyed(strategyId, keyed(hwKey, { enum: ["supported", "unsupported"] })),
      hardware_overrides: keyed(hwKey, extraArgsEnv({ install_note: { type: "string" } })),
      // looked up by exact profile id
      hardware_notes: keyed(gpuId, obj({ title: { type: "string" }, body: { type: "string" }, link: obj({ href: { type: "string" }, label: { type: "string" } }) }, { required: ["body"] })),
      strategy_overrides: keyed(strategyId, strategyOverride),
      // Tuned presets (recipe-presets.js): a named delta for one exact GPU,
      // optionally one strategy. Keys are lowercase ids; `baseline` is reserved.
      presets: (() => {
        const jsonArgs = keyed({ pattern: "^--" }, { type: "object" }, "edits to JSON-valued flags: keys to set, null deletes");
        const delta = { args: strings, env: envMap, remove_args: { type: "array", items: { type: "string", pattern: "^--" } }, json_args: jsonArgs };
        const role = obj({ ...delta, nodes: { type: "integer", minimum: 1 }, parallelism: pdMode });
        const roles = obj({ prefill: role, decode: role });
        const select = obj({
          pd_router: { enum: ["vllm", "dynamo", "smg"] },
          frontend: { enum: frontends },
          features: keyed({ type: "string" }, { anyOf: [{ type: "boolean" }, { type: "string" }] }, "true/false toggles a feature; a string picks that mode (and turns it on)"),
          dcp: { type: "integer", minimum: 1 },
          kv_offload: { anyOf: [{ type: "string" }, { const: false }] },
        });
        const image = { anyOf: [
          { type: "string" },
          obj({ vllm: { type: "string" }, dynamo: { type: "string" }, pip: strings, model_revision: { type: "string" } }),
        ] };
        const point = obj({
          label: { type: "string" },
          concurrency: { type: "integer", minimum: 1 },
          description: { type: "string" },
          default: { type: "boolean" },
          ...delta,
          roles,
          select,
          image,
        }, { required: ["label"] });
        return keyed({ pattern: "^[a-z0-9][a-z0-9_]*$" }, obj({
          label: { type: "string" },
          hardware: gpuId,
          strategy: strategyId,
          variants: { type: "array", minItems: 1, items: { type: "string" } },
          workload: { type: "string" },
          description: { type: "string" },
          default: { type: "boolean" },
          ...delta,
          roles,
          select,
          image,
          points: keyed({ pattern: "^[a-z0-9][a-z0-9_]*$" }, point),
          source: obj({ label: { type: "string" }, href: { type: "string" }, key: { type: "string" }, revision: { type: "string" } }, { required: ["href"] }),
          runbook: { type: "string" },
        }, { required: ["label", "hardware"] }));
      })(),
      // Configuration-scoped guide entries (recipe-context.js): shown below
      // the builder when `when` matches the selection.
      notes: { type: "array", items: obj({ title: { type: "string" }, when, body: { type: "string" } }, { required: ["when", "body"] }) },
      troubleshooting: { type: "array", items: obj({ title: { type: "string" }, when, body: { type: "string" } }, { required: ["title", "body"] }) },
      advanced_options: { type: "array", items: { type: "object" } },
      omni: obj({
        serve_binary: { type: "string" },
        port: { type: "integer" },
        task_label: { type: "string" },
        task_hint: { type: "string" },
        variant_label: { type: "string" },
        variant_hint: { type: "string" },
        task_axes: { type: "array", items: obj({ id: { type: "string" }, label: { type: "string" }, options: { type: "array", items: obj({ id: { type: "string" }, label: { type: "string" } }) } }) },
        tasks: { type: "array", items: omniTask },
      }, { required: ["tasks"] }),
      guide: { type: "string" },
    },
  };
}
