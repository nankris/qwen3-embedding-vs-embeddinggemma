from pathlib import Path
import subprocess
print(subprocess.getoutput('nvidia-smi'))
for f in ['/content/install098.log','/content/install_done','/content/exp098/resume_run.log','/content/exp098/resume_pipeline.log','/content/exp098/complete.json']:
 p=Path(f);print(f,p.read_text()[-2000:] if p.exists() else 'absent')
