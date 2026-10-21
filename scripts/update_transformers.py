import subprocess
print(subprocess.getoutput('pip install --no-deps --upgrade git+https://github.com/huggingface/transformers.git@8498ed425fa01bbee7d086efdf1a6634c2962aed'))
