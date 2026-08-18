import re, statistics, pathlib, json
base=pathlib.Path('docs/doc_workspace/SGCP/artifacts/channel_model_validation_20260721')
for name in ['sgcp_default_ns3_replay_job_abs','sgcp_high_capacity_ns3_replay_job_abs']:
    text=(base/name/'ns3_stdout.log').read_text(errors='ignore')
    vals=[int(m.group(1)) for m in re.finditer(r'allocated=(\d+)', text)]
    callbacks=len(re.findall(r'\[CAM_RX_CALLBACK\]', text))
    adds=len(re.findall(r'\[MANUAL_CMD_ADD\]', text))
    rejects=len(re.findall(r'\[MANUAL_CMD_REJECT\]', text))
    print(json.dumps({'variant':name,'manual_add':adds,'callbacks':callbacks,'rejects':rejects,'consume_events':len(vals),'alloc_min':min(vals) if vals else None,'alloc_mean':round(statistics.mean(vals),2) if vals else None,'alloc_max':max(vals) if vals else None}, ensure_ascii=False))
