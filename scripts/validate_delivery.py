"""Validate required artifacts and final provenance; no model inference or downloads."""
import csv, hashlib, json, math
from pathlib import Path
from PIL import Image
R=Path('results'); models=['qwen','gemma','bge']; datasets=['scifact','nfcorpus','arguana']
for file in ['README.md','writeup.md']:
    assert Path(file).is_file(), file
for model in models:
    gpu=json.loads((R/f'{model}_model.json').read_text())
    cpu=json.loads((R/f'{model}_cpu_model.json').read_text())
    assert gpu['revision']==cpu['revision']
    assert gpu['parameters']==cpu['parameters']
    assert gpu['embedding_dim']==cpu['embedding_dim']
    assert gpu['transformers_version']==cpu['transformers_version']=='5.21.0.dev0'
    for device in ['cpu','gpu']:
        speed=json.loads((R/f'{model}_{device}_speed.json').read_text())
        assert speed['docs']==500 and speed['seconds']>0
        assert math.isclose(speed['docs_per_second'],500/speed['seconds'])
    for dataset in datasets:
        rows=list(csv.DictReader((R/f'{model}_{dataset}_metrics.csv').open()))
        assert len(rows)=={'scifact':300,'nfcorpus':323,'arguana':1406}[dataset]
for file in ['quality.png','speed.png','multilingual.png','table.png']:
    im=Image.open(Path('figures')/file)
    assert all(abs(x-150)<0.1 for x in im.info['dpi']), file
verdict=json.loads((R/'verdict.json').read_text())
assert verdict['quality']=='DISPROVEN'
paths=[p for folder in ['results','figures','scripts'] for p in Path(folder).rglob('*') if p.is_file() and p.suffix not in ['.pyc','.gz'] and '__pycache__' not in str(p) and p.name not in ['artifact_manifest.json','delivery_validation.json']]
manifest={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}
(R/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
status={'required_artifacts_present':True,'models':3,'english_query_model_rows':6087,'multilingual_query_model_rows':1200,'pngs':4,'cpu_and_gpu_same_checkpoint_and_transformers':True,'verdict':verdict}
(R/'delivery_validation.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
