import importlib.metadata as M
for p in ['transformers','sentence-transformers','torch']:print(p,M.version(p))
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print('embedding_gemma2 support',CONFIG_MAPPING_NAMES.get('embedding_gemma2'))
from pathlib import Path
print(Path('/content/embedding-comparison/run.log').read_text()[-2000:])
