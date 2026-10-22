import json,datetime
from pathlib import Path
Path('results/session_stop.json').write_text(json.dumps({'stop_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'session':'embedding-comparison','command':'colab stop -s embedding-comparison'},indent=2))
