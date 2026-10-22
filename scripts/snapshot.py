from pathlib import Path
import tarfile
root=Path('/content/embedding-comparison')
with tarfile.open('/content/embedding-comparison_snapshot.tar.gz','w:gz') as t:
 for p in root.rglob('*'):
  if p.is_file() and p.suffix!='.npy':t.add(p,arcname=str(p.relative_to(root)))
print('snapshot bytes',Path('/content/embedding-comparison_snapshot.tar.gz').stat().st_size)
