import assert from "node:assert/strict";
import test from "node:test";
import { buildDockerRun, buildDockerArgv, buildAscendContainerSetup, computeDockerMeta } from "../src/lib/command-synthesis.js";

const meta = { image: "quay.io/ascend/vllm-ascend:v0.23.0-a5", isNpu: true, npuDeviceCount: 8, gpuFlags: "--device /dev/davinci0" };
const argv = ["vllm", "serve", "Eco-Tech/model", "--tensor-parallel-size", "8"];

test("two-step container setup is opt-in for the selected Ascend variant", () => {
  const profile = { generation: "npu", gpu_count: 8 };
  const variant = { hardware_overrides: { ascend_950dt: { docker_image: meta.image, install: { pip: false, docker: { container_shell: true } } } } };
  assert.equal(computeDockerMeta({}, variant, profile, "ascend_950dt").containerShell, true);
  assert.equal(computeDockerMeta({}, {}, profile, "ascend_950dt").containerShell, false);
  assert.equal(computeDockerMeta({}, variant, { generation: "hopper" }, "h200").containerShell, false);
});

test("container preparation opens bash with ModelScope cache mounted, without starting a model", () => {
  const shell = buildAscendContainerSetup(meta);
  assert(shell.includes(`docker pull ${meta.image}`));
  assert.match(shell, /docker run --rm -it/);
  assert.match(shell, /--entrypoint \/bin\/bash/);
  assert(shell.includes("~/.cache/modelscope:/root/.cache/modelscope"));
  assert(shell.endsWith(meta.image));
  assert(!shell.includes("vllm serve"));
});

test("Ascend one-shot Docker commands explicitly launch vllm serve", () => {
  const args = buildDockerArgv({ argv, env: {}, meta });
  assert.equal(args[args.indexOf("--entrypoint") + 1], "vllm");
  assert.deepEqual(args.slice(args.indexOf(meta.image) + 1), argv.slice(1));
  assert(args.includes("~/.cache/modelscope:/root/.cache/modelscope"));
  const shell = buildDockerRun({ command: argv.join(" "), env: {}, ...meta });
  assert.match(shell, /--entrypoint vllm/);
  assert(shell.includes(`${meta.image} serve Eco-Tech/model`));
});

test("NVIDIA keeps its existing vllm serve image entrypoint", () => {
  const args = buildDockerArgv({ argv, env: {}, meta: { image: "vllm/vllm-openai:latest" } });
  assert(!args.includes("--entrypoint"));
  assert.deepEqual(args.slice(args.indexOf("vllm/vllm-openai:latest") + 1), argv.slice(2));
});
