import csv, json, pathlib, statistics
base = pathlib.Path('docs/doc_workspace/SGCP/artifacts/channel_model_validation_20260721')
files = {
    'sgcp_papg_logical': base/'sgcp_papg_logical_trace.csv',
    'sgcp_papg_ns3': base/'sgcp_papg_ns3_trace.csv',
    'pcs_ns3': base/'pcs_ns3_trace.csv',
    'edgecooper_hd_ns3': base/'edgecooper_hd_ns3_trace.csv',
}
for name,path in files.items():
    rows=list(csv.DictReader(open(path,newline='')))
    comm=[float(r.get('communication_bytes') or 0) for r in rows]
    times=[float(r.get('frame_comm_time_ms') or 0) for r in rows]
    out={
        'variant': name,
        'rows': len(rows),
        'nonzero_rows': sum(1 for c in comm if c>0),
        'bytes_total': int(sum(comm)),
        'bytes_mean': round(statistics.mean(comm), 2) if comm else 0,
        'time_ms_mean': round(statistics.mean(times), 2) if times else 0,
        'time_ms_max': round(max(times), 2) if times else 0,
        'channel_estimator': sorted(set(r.get('channel_estimator','') for r in rows)),
        'num_channels': sorted(set(r.get('num_channels','') for r in rows)),
        'bandwidth_mhz': sorted(set(r.get('bandwidth_mhz','') for r in rows)),
        'tb_size': sorted(set(r.get('ns3_tb_size_bytes','') for r in rows)),
    }
    print(json.dumps(out, ensure_ascii=False))
