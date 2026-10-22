from pathlib import Path
import subprocess
print(subprocess.getoutput('nvidia-smi'))
for f in ['/content/install_embeddings.log','/content/install_done','/content/embedding-comparison/resume_run.log','/content/embedding-comparison/resume_pipeline.log','/content/embedding-comparison/complete.json']:
 p=Path(f);print(f,p.read_text()[-2000:] if p.exists() else 'absent')
