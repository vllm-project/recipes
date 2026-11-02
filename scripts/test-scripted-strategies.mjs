import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import yaml from 'js-yaml';
import { contentDir } from '../src/lib/content-root.js';
import { guideCodeBlock, resolveScriptedStrategy, resolveDeploymentScript, resolveContainerSetup } from '../src/lib/scripted-strategies.js';
import { resolveAscendPd, substituteAscendPd } from '../src/lib/ascend-pd.js';

const guide = '## Setup\n```shell\necho setup\n```\n## PD\n  ```python\n  # heading inside code\n  print(1)\n  ```\n```bash\npython launch.py\n```\n## Other\n```bash\necho wrong\n```';
const recipe = { guide, scripted_strategies: { pd_cluster: { ascend_910c: {
  label: 'Fixed PD', model_id: 'org/model', description: 'Two nodes', guide_href: '#setup',
  steps: [{ label: 'Launch', section: 'PD', block: 0, filename: 'launch.py', launch_block: 1 }],
} } } };

test('extracts only the selected section and removes fence indentation', () => {
  assert.equal(guideCodeBlock(guide, 'PD', 0), '# heading inside code\nprint(1)');
  assert.throws(() => guideCodeBlock(guide, 'PD', 2), /Missing script/);
});
test('emits quoted file content before its launch command', () => {
  const r = resolveScriptedStrategy(recipe, 'pd_cluster', 'ascend_910c');
  assert.equal(r.steps[0].command, "cat > launch.py <<'RECIPE_SCRIPT_EOF'\n# heading inside code\nprint(1)\nRECIPE_SCRIPT_EOF\n\npython launch.py");
});
test('does not override any other hardware or strategy', () => {
  assert.equal(resolveScriptedStrategy(recipe, 'pd_cluster', 'b200'), null);
  assert.equal(resolveScriptedStrategy(recipe, 'single_node_tp', 'ascend_910c'), null);
});
test('fails closed when the guide section is removed', () => {
  assert.throws(() => resolveScriptedStrategy({ ...recipe, guide: '' }, 'pd_cluster', 'ascend_910c'), /Missing script/);
});
test('does not interpolate shell variables in file content', () => {
  const r = resolveScriptedStrategy({ ...recipe, guide: guide.replace('print(1)', 'print("$HOME")') }, 'pd_cluster', 'ascend_910c');
  assert(r.steps[0].command.includes('print("$HOME")'));
  assert(r.steps[0].command.includes("<<'RECIPE_SCRIPT_EOF'"));
});
test('quotes common environment values and rejects a colliding heredoc delimiter', () => {
  const preset = recipe.scripted_strategies.pd_cluster.ascend_910c;
  const r = resolveScriptedStrategy({ ...recipe, scripted_strategies: { pd_cluster: { ascend_910c: {
    ...preset, env: { MODEL_PATH: "a'b $MODEL" },
  } } } }, 'pd_cluster', 'ascend_910c');
  assert(r.steps[0].command.startsWith("export MODEL_PATH='a'\\''b $MODEL'"));
  assert.throws(() => resolveScriptedStrategy({ ...recipe, guide: guide.replace('print(1)', 'RECIPE_SCRIPT_EOF') }, 'pd_cluster', 'ascend_910c'), /Reserved script delimiter/);
});

test('named scripts do not depend on Guide content and never fall back on missing names', () => {
  const r = { guide, deployment_scripts: { ascend_910c: { template: 'echo "$HOME"', launch: 'bash run.sh', docker: 'docker run image' } },
    container_setup: { ascend_910c: { script: 'docker', container_script: 'template' } },
    scripted_strategies: { multi_node_tp_dp: { ascend_910c: {
      steps: [{ label: 'Node 0', script: 'template', filename: 'run.sh', launch_script: 'launch' }],
    } } } };
  const expected = resolveScriptedStrategy(r, 'multi_node_tp_dp', 'ascend_910c');
  for (const guide of ['', '## Renamed\n```bash\necho unrelated\n```', recipe.guide.split('\n').reverse().join('\n')]) {
    assert.deepEqual(resolveScriptedStrategy({ ...r, guide }, 'multi_node_tp_dp', 'ascend_910c'), expected);
  }
  assert.equal(resolveContainerSetup(r, 'ascend_910c').command, 'docker run image');
  assert.equal(resolveContainerSetup(r, 'ascend_910c').containerCommand, 'echo "$HOME"');
  assert.equal(resolveContainerSetup(r, 'h200'), null);
  assert.deepEqual(resolveContainerSetup({container_setup:{h200:'#old-setup'}}, 'h200'), {guideHref:'#old-setup'});
  assert.throws(() => resolveDeploymentScript(r, 'ascend_910c', {script:'missing',section:'PD',block:0}), /Missing deployment script/);
  assert.throws(() => resolveDeploymentScript(r, 'ascend_950dt_8x', {script:'template'}), /Missing deployment script/);
});

const pdGuide = ['## Nodes', '```bash', 'nic_name="xxx"', 'local_ip="xxx"', 'vllm serve /weights/model --served-model-name test-model', '```',
  '```bash', 'bash run_dp_template.sh', '```',
  '```bash', 'python launch_online_dp.py --dp-size 8 --pp-size 1 --dp-address $node_d0_ip', '```',
  '```bash', 'python proxy.py --prefiller-hosts $node_p0_ip --decoder-hosts $node_d0_ip $node_d1_ip --port 8000', '```',
  '## Setup', '```bash', 'docker run --net=host image bash', '```',
  '```python', 'import argparse', '# --pp-size', '```'].join('\n');
const pdRecipe = { guide: pdGuide, scripted_strategies: { pd_cluster: { ascend_910c: {
  variant: 'ascend_w8a8', model_id: 'org/model', router: {id:'ascend_proxy',port:8000},
  hardware_label: 'A3', constraints: 'Fixed test topology',
  workdir: '/root/pd', source: {revision:'a'.repeat(40),url:'https://example.org/source'},
  preparation: {docker:{section:'Setup',block:0},launcher:{section:'Setup',block:1}},
  steps: [
    ...['prefill','decode'].flatMap(role => [0,1].map(node => ({label:role+node,role,node,section:'Nodes',block:0,launch_block:role==='prefill'?1:2,launch_type:role==='prefill'?'direct':'launcher'}))),
    {label:'Proxy',role:'proxy',section:'Nodes',block:3},
  ],
} } } };

test('Ascend PD separates file saving from per-role launch commands', () => {
  const out=resolveAscendPd(pdRecipe,'ascend_w8a8','ascend_910c',{prefill:{rank:1},decode:{rank:1}});
  assert.equal(out.router.label,'Ascend Proxy');
  assert.equal(out.prefill.label,'P1');
  assert(out.prefill.template.includes('local_ip="<P1_IP>"'));
  assert(out.prefill.launchCommand.endsWith('bash run_dp_template.sh'));
  assert(!out.prefill.command.includes('bash run_dp_template.sh'));
  assert(out.decode.launchCommand.includes('python launch_online_dp.py'));
  assert(!out.decode.template.includes('python launch_online_dp.py'));
  assert(out.decode.template.includes('nic_name="<D1_NIC>"'));
  assert(out.decode.launchCommand.includes('"<D0_IP>"'));
  assert.equal(out.servedModelName,'test-model');
  assert.equal(out.preparation.launcherSource,'import argparse\n# --pp-size');
  assert.equal(out.preparation.launcherPath,'/root/pd/launch_online_dp.py');
  assert.equal(out.preparation.hardwareLabel,'A3');
  assert.deepEqual(out.preparation.launcherNodes,['D0','D1']);
  assert.deepEqual(out.preparation.directNodes,['P0','P1']);
  assert(out.decode.launchCommand.includes('First save the common launcher'));
  assert(!out.prefill.launchCommand.includes('First save the common launcher'));
  assert(!('containerCommand' in out.preparation));
  assert.equal(substituteAscendPd('local_ip="<D1_IP>"',{}),'local_ip="<D1_IP>"');
  assert.equal(substituteAscendPd('local_ip="<D1_IP>"',{DECODE_NODE_2:'192.0.2.4'}),'local_ip="192.0.2.4"');
});
test('Ascend PD is explicitly configured and rejects another variant', () => {
  assert.equal(resolveAscendPd(pdRecipe,'ascend_w8a8','h200'),null);
  assert.equal(resolveAscendPd(pdRecipe,'ascend_w8a8','ascend_950dt_8x'),null);
  assert.throws(()=>resolveAscendPd(pdRecipe,'fp8','ascend_910c'),/requires variant/);
});
test('Ascend PD validates node indices and keeps launch method per node', () => {
  const bad=structuredClone(pdRecipe);
  bad.scripted_strategies.pd_cluster.ascend_910c.steps[1].node=3;
  assert.throws(()=>resolveAscendPd(bad,'ascend_w8a8','ascend_910c'),/Invalid Ascend/);
  const next=structuredClone(pdRecipe);
  Object.assign(next.scripted_strategies.pd_cluster.ascend_910c.steps[0],{launch_type:'launcher',launch_block:2});
  assert(resolveAscendPd(next,'ascend_w8a8','ascend_910c').prefill.launchCommand.includes('python launch_online_dp.py'));
  for(const step of next.scripted_strategies.pd_cluster.ascend_910c.steps.filter(s=>s.role==='prefill')) Object.assign(step,{launch_type:'launcher',launch_block:2});
  const both=resolveAscendPd(next,'ascend_w8a8','ascend_910c');
  assert.deepEqual(both.preparation.launcherNodes,['P0','P1','D0','D1']);
  assert.deepEqual(both.preparation.directNodes,[]);
});

test('named Ascend preparation, role templates, launches and proxy are Guide-independent', () => {
  const r = structuredClone(pdRecipe), preset = r.scripted_strategies.pd_cluster.ascend_910c;
  r.deployment_scripts = {ascend_910c:{}};
  const add = (name, ref) => { r.deployment_scripts.ascend_910c[name] = guideCodeBlock(r.guide, ref.section, ref.block); return name; };
  for (const [i, step] of preset.steps.entries()) {
    step.script = add(`node_${i}`, step);
    if (step.launch_block !== undefined) step.launch_script = add(`launch_${i}`, {section:step.section,block:step.launch_block});
    delete step.section; delete step.block; delete step.launch_block;
  }
  for (const [key, ref] of Object.entries(preset.preparation)) preset.preparation[key] = {script:add(key,ref)};
  const expected = resolveAscendPd(pdRecipe,'ascend_w8a8','ascend_910c');
  for (const guide of ['', '## Different\n```python\nprint("unrelated")\n```']) {
    assert.deepEqual(resolveAscendPd({...r,guide},'ascend_w8a8','ascend_910c'), expected);
  }
  delete r.deployment_scripts.ascend_910c.launcher;
  assert.throws(() => resolveAscendPd(r,'ascend_w8a8','ascend_910c'), /Missing deployment script/);
});

test('assembled recipes with named scripts survive arbitrary Guide edits and source round-trip', () => {
  const files = dir => fs.readdirSync(dir, {withFileTypes:true}).flatMap(e => e.isDirectory()
    ? files(path.join(dir,e.name)) : /\.ya?ml$/.test(e.name) ? [path.join(dir,e.name)] : []);
  for (const file of files(path.join(contentDir(),'models'))) {
    const recipe = yaml.load(fs.readFileSync(file,'utf8'));
    if (!recipe.deployment_scripts) continue;
    const roundTrip = yaml.load(yaml.dump(recipe,{noRefs:true,lineWidth:-1}));
    const guides = ['', '## Renamed deployment\nUnrelated explanation.\n```bash\necho unrelated\n```',
      recipe.guide.split('\n').reverse().join('\n')];
    for (const [strategy, hardware] of Object.entries(recipe.scripted_strategies || {})) {
      for (const [hw, preset] of Object.entries(hardware)) {
        assert(preset.steps.every(s => s.script && s.block === undefined && s.launch_block === undefined), `${file}/${hw}: positional script reference`);
        const render = r => preset.router?.id === 'ascend_proxy'
          ? resolveAscendPd(r,preset.variant,hw) : resolveScriptedStrategy(r,strategy,hw);
        const expected = render(recipe);
        assert.deepEqual(render(roundTrip),expected,file);
        for (const guide of guides) assert.deepEqual(render({...recipe,guide}),expected,file);
        if (preset.router?.id === 'ascend_proxy') for (const node of [0,1]) {
          const pdNodes = {prefill:{rank:node},decode:{rank:node}};
          assert.deepEqual(resolveAscendPd({...recipe,guide:''},preset.variant,hw,pdNodes),resolveAscendPd(recipe,preset.variant,hw,pdNodes),file);
        }
      }
    }
    for (const hw of Object.keys(recipe.container_setup || {})) for (const guide of guides) {
      assert.deepEqual(resolveContainerSetup({...recipe,guide},hw),resolveContainerSetup(recipe,hw),file);
    }
  }
});
