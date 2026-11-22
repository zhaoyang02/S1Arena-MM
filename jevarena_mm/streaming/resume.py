"""Reuse the earliest successful episode, retaining source traces and runtime strata."""
import json
import shutil
from .manifest import digest


def reuse(sources, progress, out):
    selected = set(progress['case_ids'])
    rows, seen, strata = [], set(), []
    for source in sources:
        path = source / 'progress.json'
        old = json.loads(path.read_text())
        for key in ('protocol', 'manifest_sha256', 'model', 'backend', 'config'):
            if old.get(key) != progress.get(key):
                raise ValueError(f'Resume identity mismatch: {key} in {source}')
        ids = []
        for row in old['results']:
            ident = row['id']
            if ident not in selected or ident in seen or row['status'] != 'completed':
                continue
            trace_dir = source / ident
            events = [json.loads(line) for line in (trace_dir / 'events.jsonl').read_text().splitlines() if line.strip()]
            if not any(e['event'] == 'observation_end' for e in events):
                raise ValueError(f'Incomplete trace for successful episode: {ident}')
            shutil.copytree(trace_dir, out / ident)
            rows.append(row)
            seen.add(ident)
            ids.append(ident)
        strata.append({'progress_sha256': digest(path), 'case_ids': ids,
                       'code_sha256': old.get('code_sha256', {}), 'runtime': old.get('runtime', {})})
    progress['analysis'] = {'resume_policy': 'first successful result, supplied attempt order; no score selection',
                            'reused_runtime_strata': strata,
                            'new_runtime_case_ids': [i for i in progress['case_ids'] if i not in seen],
                            'timing_scope': 'mixed runtime versions; report strata separately'}
    return rows
