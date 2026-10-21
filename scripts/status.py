from pathlib import Path
import subprocess
print(subprocess.getoutput('nvidia-smi'))
for name in ['run.log','mini.log','pipeline.log']:
 p=Path('/content/exp098')/name
 print(name, p.read_text()[-4500:] if p.exists() else 'absent')
print([p.name for p in Path('/content/exp098/results').glob('*')])
