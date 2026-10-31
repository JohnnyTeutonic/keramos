import hashlib,json,pathlib,subprocess
a=pathlib.Path('/home/jonat/keramos_routed_actual_parent_20261031')
root=pathlib.Path('/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/keramos')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=a/'target/release/kcr'
new=pathlib.Path('/home/jonat/cleanroom_build/target_codex_20261030/release/kcr')
fixture=a/'current/conformance/state/routed_longrope_session_length.json'
gguf=a/'current/conformance/regressions/routed_longrope/routed_longrope.gguf'
d={'old_source_commit':(a/'OLD_COMMIT').read_text().strip(),'current_runner_fixture_commit':(a/'CURRENT_COMMIT').read_text().strip(),'ops_fix_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','1fd77ed'],text=True).strip(),'build':{'offline':True,'jobs':1,'rustflags':'-C codegen-units=1','address_space_limit_bytes':1610612736,'budget_session':'keramos','cargo':subprocess.check_output(['/home/jonat/.cargo/bin/cargo','--version'],text=True).strip(),'rustc':subprocess.check_output(['/home/jonat/.cargo/bin/rustc','--version'],text=True).strip(),'note':'First archive-only build failed: schema.rs requires ../../ir.schema.json. Added unmodified schema from SAME old commit; successful retry retained all resource caps.'},'old_source_sha256':{str(p.relative_to(a/'old')):sha(p) for p in sorted((a/'old').rglob('*')) if p.is_file()},'current_runner_modules_sha256':{str(p.relative_to(a/'current')):sha(p) for p in sorted((a/'current/tools').glob('*.py'))},'fixtures':{str(p.relative_to(a/'current')):sha(p) for p in [fixture,gguf,a/'current/conformance/manifest.json']},'binaries':{'old':{'path':str(old),'sha256':sha(old)},'corrected':{'path':str(new),'sha256':sha(new),'provenance_limit':'Pre-existing corrected binary supplied by parent. Hash pinned here; not rebuilt for this paired check.'}},'results':{k:json.loads((a/'runs'/f'{k}.json').read_text()) for k in ['old','corrected']},'exit_codes':{k:int((a/'runs'/f'{k}.exit').read_text()) for k in ['old','corrected']}}
assert d['results']['old']['counts']['FAIL']==1
assert d['results']['corrected']['counts']['PASS']==1
assert d['exit_codes']=={'old':1,'corrected':0}
with (a/'audit.json').open('x') as f:json.dump(d,f,indent=1);f.write('\n')
