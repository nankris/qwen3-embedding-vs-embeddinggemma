import json,datetime
from pathlib import Path
Path('results/session_stop.json').write_text(json.dumps({'stop_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'session':'b6-098','command':'colab stop -s b6-098'},indent=2))
