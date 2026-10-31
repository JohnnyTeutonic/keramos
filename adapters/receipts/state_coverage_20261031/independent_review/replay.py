from pathlib import Path
import hashlib, importlib.util, json, platform, sys, tempfile

root = Path(r'C:/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos')
out = Path(r'C:/Users/jonat/AppData/Local/Temp/window_rewind_review_20261031')
out.mkdir(exist_ok=True)
sys.path.insert(0, str(root / 'tools'))
import numpy as np
import ir_conformance as conf
import ir_conformance_gen as gen
import ir_window_rewind_coverage as window
import ir_routed_longrope_coverage as routed
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = ['tools/ir_conformance_gen.py', 'tools/ir_conformance.py', 'tools/ir_numpy.py', 'tools/ir_window_rewind_coverage.py', 'tools/ir_routed_longrope_coverage.py', 'tests/test_window_rewind_conformance.py', 'conformance/manifest.json']
before = {p: sha(root / p) for p in sources}
original = gen.SUITE
comparisons = []
with tempfile.TemporaryDirectory(prefix='keramos-targeted-regen-') as temp:
    tmp = Path(temp)
    gen.SUITE = tmp
    load = conf.load_model
    conf.load_model = lambda path: load(original / 'models' / 'pk_mor')
    rcases = routed.routed_longrope_traces()
    conf.load_model = load
    # Window generator is read-only: redirect only its model lookup to originals.
    window.SUITE = original
    cases = rcases + window.discarded_window_traces()
    (tmp / 'state').mkdir()
    for name, case in cases:
        dst = tmp / 'state' / (name + '.json')
        dst.write_text(json.dumps(case, indent=1) + '\n', encoding='utf-8')
        actual = original / 'state' / dst.name
        comparisons.append({'path': 'state/' + dst.name, 'generated_sha256': sha(dst), 'committed_sha256': sha(actual), 'bytes_equal': dst.read_bytes() == actual.read_bytes()})
    for generated in sorted((tmp / 'regressions').rglob('*')):
        if generated.is_file():
            relative = generated.relative_to(tmp)
            actual = original / relative
            comparisons.append({'path': relative.as_posix(), 'generated_sha256': sha(generated), 'committed_sha256': sha(actual), 'bytes_equal': generated.read_bytes() == actual.read_bytes()})
    manifest = json.loads((original / 'manifest.json').read_text())
    entries = {v['id']: v for v in manifest['cases']} if isinstance(manifest, dict) else {v['id']:v for v in manifest}
    manifest_proofs = []
    for name, case in cases:
        entry = entries['trace/' + name]
        expected = {'id': 'trace/' + name, 'kind':'trace', 'file':'state/' + name + '.json', 'model':case['model'], 'profile':'state'}
        manifest_proofs.append({'id': entry['id'], 'exact_generic_regeneration_entry': entry == expected})
    gen.SUITE = original
spec = importlib.util.spec_from_file_location('bounded_window_test', root / 'tests/test_window_rewind_conformance.py')
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
evidence=[]
case=window.discarded_window_traces()[0][1]
for mode in ('retain', 'safe', 'unsafe', 'refuse'):
    adapter=test.BoundedKV(mode)
    try:
        ok,message=conf.run_trace(adapter,case,original/case['model'],1e-5)
        result={'numeric_pass':bool(ok),'message':message}
    except NotImplementedError as exc:
        result={'numeric_pass':False,'explicit_unsupported_refusal':str(exc)}
    result.update(mode=mode, actual_cache_evidence=adapter.evidence)
    evidence.append(result)
after={p:sha(root/p) for p in sources}
report={'status':'PASS' if all(v['bytes_equal'] for v in comparisons) and all(v['exact_generic_regeneration_entry'] for v in manifest_proofs) and before==after else 'FAIL', 'platform':platform.platform(),'python':sys.version,'interpreter':sys.executable,'numpy':np.__version__,'source_sha256':before,'source_unchanged':before==after,'targeted_regeneration':comparisons,'manifest_regeneration':manifest_proofs,'bounded_cache_controls':evidence,'scope':'Targeted three-case regeneration only; no real suite regeneration. Test adapter physically discards K/V arrays but is not evidence that all external engines evict. Retention or explicit unsupported refusal remains legal. No model training, GPU, provider or config operation.'}
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['status']=='PASS', report['targeted_regeneration']
assert evidence[0]['numeric_pass'] and evidence[1]['numeric_pass'] and not evidence[2]['numeric_pass'] and 'explicit_unsupported_refusal' in evidence[3]
print(json.dumps({'status':report['status'],'comparisons':len(comparisons),'output':str(out),'modes':[{k:v for k,v in x.items() if k!='actual_cache_evidence'} for x in evidence]},indent=2))
