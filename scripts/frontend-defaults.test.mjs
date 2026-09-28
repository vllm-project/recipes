import assert from 'node:assert/strict';
import test from 'node:test';
import { resolveFrontend, resolveCommand } from '../src/lib/command-synthesis.js';

const recipe = {
  model: { model_id: 'example/model', default_frontend: { default: 'rust', ascend_950dt: 'python' } },
  variants: { default: { precision: 'bf16', vram_minimum_gb: 1 } },
};

test('hardware defaults preserve the fallback and explicit frontend choices', () => {
  assert.equal(resolveFrontend(recipe, undefined, 'h200'), 'rust');
  assert.equal(resolveFrontend(recipe, undefined, 'ascend_950dt'), 'python');
  assert.equal(resolveFrontend(recipe, 'python', 'h200'), 'python');
  assert.equal(resolveFrontend(recipe, 'rust', 'ascend_950dt'), 'rust');
});

test('legacy string and absent defaults retain their behavior', () => {
  assert.equal(resolveFrontend({ model: { default_frontend: 'rust' } }), 'rust');
  assert.equal(resolveFrontend({}), 'python');
});

test('synthesized commands use the hardware default without a frontend selection', () => {
  const strategies = { single_node_tp: { deploy_type: 'single_node', parallelism: 'tp' } };
  const taxonomy = { hardware_profiles: {
    h200: { brand: 'NVIDIA', gpu_count: 8, vram_gb: 141 },
    ascend_950dt: { brand: 'Huawei', gpu_count: 8, vram_gb: 96 },
  } };
  const command = hw => resolveCommand(recipe, 'default', 'single_node_tp', hw, [], strategies, taxonomy);
  assert.equal(command('h200').env.VLLM_USE_RUST_FRONTEND, '1');
  assert.equal(command('ascend_950dt').env.VLLM_USE_RUST_FRONTEND, undefined);
});
