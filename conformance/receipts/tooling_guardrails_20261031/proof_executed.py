import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path('/home/jonat/codex_worktrees/keramos-conformance-20261031')
os.chdir(ROOT)
R = ROOT / 'conformance/receipts/tooling_guardrails_20261031'
source = ROOT / 'tools/ir_conformance.py'
base = subprocess.check_output(['git', 'rev-parse', 'main'], text=True).strip()
old = subprocess.check_output(['git', 'show', base + ':tools/ir_conformance.py'])
new = source.read_bytes()
negative = ['python3', '-m', 'pytest', 'tests/test_ir_conformance.py', '-q',
            '-k', 'one_sided or nonfinite_comparison or matching_nan or matching_infinity or expected_state or required_state or without_an_expectation or independent_recurrent or model_case or model_score']
if (R / 'negative_before.log').exists():
    (R / 'negative_before.log').rename(R / 'negative_initial_fixture.log')
try:
    source.write_bytes(old)
    with (R / 'negative_before.log').open('w') as log:
        result = subprocess.run(negative, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 1, result.returncode
finally:
    source.write_bytes(new)

commands = [
    ('negative_after', negative),
    ('pytest_after', ['python3', '-m', 'pytest', 'tests', '-q']),
    ('ruff', ['python3', '-m', 'ruff', 'check', 'tools/ir_conformance.py', 'tests/test_ir_conformance.py']),
    ('reference_after', ['python3', 'tools/ir_conformance.py', '--adapter', 'ir_numpy',
                         '--json', str(R / 'reference_after.json')]),
    ('cleanroom_after', ['python3', 'tools/ir_conformance.py', '--adapter',
                         'cmd:/home/jonat/codex_worktrees/keramos-conformance-build/release/kcr --conformance',
                         '--json', str(R / 'cleanroom_after.json')]),
]
for name, command in commands:
    with (R / (name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, (R / (name + '.log')).read_text().splitlines()[-1], flush=True)

comparisons = {}
for name, historic in [('reference_after', 'ir_numpy.json'), ('cleanroom_after', 'cleanroom.json')]:
    current = json.loads((R / (name + '.json')).read_text())
    previous = json.loads((ROOT / 'conformance/receipts' / historic).read_text())
    actual = {v['id']: v['verdict'] for v in current['results']}
    expected = {v['id']: v['verdict'] for v in previous['results']}
    assert actual == expected, name
    comparisons[name] = {'counts': current['counts'], 'verdicts_identical_to': 'conformance/receipts/' + historic,
                         'compared_cases': len(actual)}

subprocess.run(['git', 'diff', '--exit-code', base, '--', 'IR_SPEC.md', 'ir.schema.json',
                'conformance/ops', 'conformance/models', 'conformance/state', 'conformance/refuse',
                'conformance/accept', 'conformance/digest', 'conformance/quant', 'conformance/grad',
                'conformance/chat', 'cleanroom', 'tools/ir_numpy.py', 'conformance/manifest.json',
                'conformance/receipts/ir_numpy.json', 'conformance/receipts/cleanroom.json'], check=True)
subprocess.run(['git', 'diff', '--check'], check=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
files = ['tools/ir_conformance.py', 'tests/test_ir_conformance.py', 'conformance/README.md']
receipt = {
    'base_commit': base,
    'scope': 'Conformance-tool correctness only; no normative statement/schema/reference arithmetic/expected values changed',
    'nonfinite_policy': 'matching NaN masks allowed; unmatched NaNs fail; infinity-bearing ROUNDING comparison cannot establish pass; EXACT unchanged',
    'state_policy': 'only fixture-required state is checked; missing slot/tensor fails; extra or unrequested reports ignored',
    'environment': {'python': platform.python_version(), 'platform': platform.platform()},
    'sources': {name: sha(ROOT / name) for name in files},
    'pre_fix_runner_sha256': hashlib.sha256(old).hexdigest(),
    'test_source_same_for_before_and_after': sha(ROOT / 'tests/test_ir_conformance.py'),
    'rust_binary_sha256': sha('/home/jonat/codex_worktrees/keramos-conformance-build/release/kcr'),
    'replays': comparisons,
    'evidence': [p.name for p in sorted(R.iterdir()) if p.name != 'validation.json'],
}
(R / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
