import csv, json, pathlib, statistics, re
base = pathlib.Path('docs/doc_workspace/SGCP/artifacts/channel_model_validation_20260721')
trace_files = {
    'SGCP-PAPG logical estimator': base/'sgcp_papg_logical_trace.csv',
    'SGCP-PAPG NS3 estimator': base/'sgcp_papg_ns3_trace.csv',
    'PCS NS3 estimator': base/'pcs_ns3_trace_rerun.csv',
    'EdgeCooper-HD NS3 estimator': base/'edgecooper_hd_ns3_trace.csv',
}
trace_rows=[]
for name,path in trace_files.items():
    rows=list(csv.DictReader(open(path,newline='')))
    comm=[float(r.get('communication_bytes') or 0) for r in rows]
    times=[float(r.get('frame_comm_time_ms') or 0) for r in rows]
    trace_rows.append({
        'variant': name,
        'rows': len(rows),
        'nonzero_rows': sum(c>0 for c in comm),
        'bytes_total': int(sum(comm)),
        'bytes_mean': round(statistics.mean(comm), 2) if comm else 0,
        'frame_time_ms_mean': round(statistics.mean(times), 2) if times else 0,
        'frame_time_ms_max': round(max(times), 2) if times else 0,
        'channel_estimator': ','.join(sorted(set(r.get('channel_estimator','') for r in rows))),
        'num_channels': ','.join(sorted(set(r.get('num_channels','') for r in rows))),
        'bandwidth_mhz': ','.join(sorted(set(r.get('bandwidth_mhz','') for r in rows))),
        'ns3_tb_size_bytes': ','.join(sorted(set(r.get('ns3_tb_size_bytes','') for r in rows))),
    })
with open(base/'opencda_estimator_summary.csv','w',newline='') as f:
    fieldnames=list(trace_rows[0].keys())
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(trace_rows)

replay_names = [
    ('NS3 default', 'sgcp_default_ns3_replay_job_abs', 'default MCS20 symbols9 PSCCH10 RRI5'),
    ('NS3 invalid PSCCH/RRI high-capacity probe', 'sgcp_high_capacity_ns3_replay_job_abs', 'slPscchRbs=4 invalid'),
    ('NS3 invalid RRI high-capacity probe', 'sgcp_high_mcs_symbols_ns3_replay_job_abs', 'RRI1 violates resource selection window'),
    ('NS3 high MCS/symbols', 'sgcp_high_mcs_symbols_rri5_ns3_replay_job_abs', 'MCS28 symbols12 PSCCH10 RRI5'),
]
replay_rows=[]
for label,name,params in replay_names:
    path=base/name/'ns3_stdout.log'
    if not path.exists():
        continue
    text=path.read_text(encoding='utf-16', errors='ignore')
    vals=[int(m.group(1)) for m in re.finditer(r'allocated=(\d+)', text)]
    delays=[int(m.group(1))-int(m.group(2)) for m in re.finditer(r'"receive_timestamp":(\d+),"send_timestamp":(\d+)', text)]
    row={
        'variant': label,
        'params': params,
        'fatal': int('NS_FATAL' in text or 'SIGABRT' in text or 'Invalid number of RBs' in text),
        'manual_cmd_add': len(re.findall(r'\[MANUAL_CMD_ADD\]', text)),
        'cam_received': len(delays),
        'consume_events': len(vals),
        'allocated_mean_bytes': round(statistics.mean(vals), 2) if vals else '',
        'allocated_min_bytes': min(vals) if vals else '',
        'allocated_max_bytes': max(vals) if vals else '',
        'delay_mean_ms': round(statistics.mean(delays), 2) if delays else '',
        'delay_p95_ms': sorted(delays)[int(len(delays)*0.95)-1] if delays else '',
        'delay_max_ms': max(delays) if delays else '',
    }
    replay_rows.append(row)
with open(base/'ns3_replay_summary.csv','w',newline='') as f:
    fieldnames=list(replay_rows[0].keys())
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(replay_rows)

md = [
    '# Channel Model Validation 2026-07-21',
    '',
    '## OpenCDA estimator smoke',
    '',
    '| Variant | Rows | Nonzero rows | Total bytes | Mean frame time ms | Max frame time ms | Estimator | Bandwidth MHz | TB bytes |',
    '|---|---:|---:|---:|---:|---:|---|---:|---:|',
]
for r in trace_rows:
    md.append('| {variant} | {rows} | {nonzero_rows} | {bytes_total} | {frame_time_ms_mean} | {frame_time_ms_max} | {channel_estimator} | {bandwidth_mhz} | {ns3_tb_size_bytes} |'.format(**r))
md += [
    '',
    '## NS3 replay',
    '',
    '| Variant | Params | Fatal | Manual adds | CAM received | Consume events | Alloc mean B | Delay mean ms | Delay P95 ms | Delay max ms |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in replay_rows:
    md.append('| {variant} | {params} | {fatal} | {manual_cmd_add} | {cam_received} | {consume_events} | {allocated_mean_bytes} | {delay_mean_ms} | {delay_p95_ms} | {delay_max_ms} |'.format(**r))
md += [
    '',
    '## Interpretation',
    '',
    '- The shared `ChannelModel` metadata is present in SGCP, PCS, and EdgeCooper/Selective traces.',
    '- `logical` and `ns3` estimator modes materially change SGCP/PAPG grid budget and estimated frame time; this validates that the scheduler path is using the unified estimator, not only writing metadata.',
    '- Default NS3 still serves about 400 B per grant and gives SGCP chunked replay P95/max delay of 110/123 ms.',
    '- Raising MCS and PSSCH symbols while keeping legal PSCCH/RRI settings raises grant size to about 899 B and reduces P95/max delay to 54/55 ms for the same 82 chunks.',
    '- `slPscchRbs=4` is invalid in the current pool factory, and `slRriMs=1` violates the resource-selection-window constraint. These are not safe capacity knobs without deeper pool-window changes.',
]
(base/'VALIDATION_SUMMARY.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))
