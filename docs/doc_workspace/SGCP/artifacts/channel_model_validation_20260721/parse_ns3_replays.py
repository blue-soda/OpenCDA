import re, statistics, pathlib, json
base=pathlib.Path('docs/doc_workspace/SGCP/artifacts/channel_model_validation_20260721')
for name in ['sgcp_default_ns3_replay_job_abs','sgcp_high_mcs_symbols_ns3_replay_job_abs']:
    path=base/name/'ns3_stdout.log'
    # PowerShell redirection stores UTF-16LE for Start-Job output.
    text=path.read_text(encoding='utf-16', errors='ignore')
    vals=[int(m.group(1)) for m in re.finditer(r'allocated=(\d+)', text)]
    rx=[int(m.group(1))-int(m.group(2)) for m in re.finditer(r'"receive_timestamp":(\d+),"send_timestamp":(\d+)', text)]
    out={'variant':name,'manual_add':len(re.findall(r'\[MANUAL_CMD_ADD\]', text)),'cam_received':len(rx),'consume_events':len(vals)}
    if vals:
        out.update({'allocated_mean':round(statistics.mean(vals),2),'allocated_min':min(vals),'allocated_max':max(vals)})
    if rx:
        out.update({'delay_mean_ms':round(statistics.mean(rx),2),'delay_p95_ms':sorted(rx)[int(len(rx)*0.95)-1],'delay_max_ms':max(rx)})
    print(json.dumps(out, ensure_ascii=False))
