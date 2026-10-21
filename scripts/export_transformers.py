from pathlib import Path
import shutil
p=next(Path('/root/.cache/pip/wheels').rglob('transformers-5.21.0.dev0-py3-none-any.whl'))
shutil.copy(p,'/content/transformers-5.21.0.dev0-py3-none-any.whl')
print('exported wheel',p)
