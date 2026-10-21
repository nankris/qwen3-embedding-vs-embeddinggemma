from pathlib import Path
import tarfile
root=Path('/content/exp098')
with tarfile.open('/content/exp098_snapshot.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix!='.npy':t.add(p,arcname=str(p.relative_to(root)))
print('snapshot bytes',Path('/content/exp098_snapshot.tar.gz').stat().st_size)
