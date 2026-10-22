import subprocess
from pathlib import Path
script="""import subprocess
p=subprocess.run(['pip','install','--no-deps','--upgrade','git+https://github.com/huggingface/transformers.git@8498ed425fa01bbee7d086efdf1a6634c2962aed'],check=True)
open('/content/install_done','w').write('ok')
"""
Path('/content/install_embeddings.py').write_text(script)
p=subprocess.Popen(['python','-u','/content/install_embeddings.py'],stdout=open('/content/install_embeddings.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print('installer PID',p.pid)
