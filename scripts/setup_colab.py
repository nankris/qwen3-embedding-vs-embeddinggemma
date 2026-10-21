import subprocess
print(subprocess.getoutput('nvidia-smi'))
print(subprocess.getoutput('pip -q install -U sentence-transformers datasets'))
