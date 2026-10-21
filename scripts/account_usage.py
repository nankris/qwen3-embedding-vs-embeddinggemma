"""Compute attributable CU bounds; account-wide balance includes other users' work."""
import json,datetime
from pathlib import Path
r=Path('results');events=[json.loads(x) for x in (r/'resume_colab_history.jsonl').read_text().splitlines()]
parse=lambda s:datetime.datetime.fromisoformat(s)
starts=[x for x in events if x['event_type']=='session_created']
assert len(starts)==2
old_start=parse(starts[0]['timestamp']);new_start=parse(starts[1]['timestamp'])
pruned=next(x for x in events if x['event_type']=='session_terminated' and x.get('reason')=='pruned')
old_last=max(parse(x['timestamp']) for x in events if x['event_type'] in ['execution','file_operation'] and parse(x['timestamp'])<parse(pruned['timestamp']))
end=json.loads((r/'session_stop.json').read_text())['stop_utc'];new_end=parse(end)
rate=1.54
old_min=(old_last-old_start).total_seconds()/3600*rate;old_max=(parse(pruned['timestamp'])-old_start).total_seconds()/3600*rate
new=(new_end-new_start).total_seconds()/3600*rate
out={'gpu':'NVIDIA L4','rate_cu_per_hour':rate,'interrupted_start_utc':old_start.isoformat(),'interrupted_last_observed_utc':old_last.isoformat(),'interrupted_pruned_utc':pruned['timestamp'],'interrupted_cu_bounds':[old_min,old_max],'resumed_start_utc':new_start.isoformat(),'resumed_stop_utc':end,'resumed_lifetime_estimated_cu':new,'total_cu_bounds':[old_min+new,old_max+new],'limitation':'Lifetime × marginal rate estimates, not exact billing. Old termination time unavailable; pruning timestamp is an upper bound. Account-wide balance changes are not attributable to this experiment.'}
(r/'compute_usage.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
