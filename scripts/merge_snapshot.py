"""Merge downloaded GPU archive without overwriting CPU metadata or timing."""
from pathlib import Path
import tarfile,shutil
root=Path.cwd();stage=root/'.resume_snapshot';stage.mkdir(exist_ok=True)
with tarfile.open(root/'results/resume_snapshot.tar.gz') as t:t.extractall(stage,filter='data')
for p in stage.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(stage);target=root/rel
 if rel.parts[0]=='results' and p.name.endswith('_model.json') and len(rel.parts)==2:
  target=root/'results/final_gpu_models'/p.name
 target.parent.mkdir(exist_ok=True,parents=True);shutil.copy(p,target)
shutil.rmtree(stage)
print('GPU archive merged; final GPU metadata retained separately until CPU completion')
