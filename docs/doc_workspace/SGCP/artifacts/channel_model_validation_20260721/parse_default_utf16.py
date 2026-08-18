import re, statistics, pathlib, json
path=pathlib.Path('docs/doc_workspace/SGCP/artifacts/channel_model_validation_20260721/sgcp_default_ns3_replay_job_abs/ns3_stdout.log')
text=path.read_text(encoding='utf-16', errors='ignore')
vals=[int(m.group(1)) for m in re.finditer(r'allocated=(\d+)', text)]
rx=[]
for m in re.finditer(r'"receive_timestamp":(\d+),"send_timestamp":(\d+)', text):
    rx.append(int(m.group(1))-int(m.group(2)))
print(json.dumps({'allocated_events':len(vals),'allocated_mean':round(statistics.mean(vals),2),'allocated_min':min(vals),'allocated_max':max(vals),'cam_received':len(rx),'delay_mean':round(statistics.mean(rx),2),'delay_p95':sorted(rx)[int(len(rx)*0.95)-1],'delay_max':max(rx)}, ensure_ascii=False))
