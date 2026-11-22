"""Run one audio model with a cached smoke gate, bounded batches and live coverage.

Use the same spec in sweeps/audio_baselines.toml. Every row comes from this harness's own calls.
Successful cached calls are skipped by the shared runner. A technical batch failure stops the model.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jevarena_mm import dataio, runner, sweep
from jevarena_mm.registry import make_model, tasks_under


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--status', type=Path, required=True)
    args = p.parse_args()
    if args.spec.split('@', 1)[0] in {
        'lm:Qwen/Qwen2.5-Omni-7B-AWQ', 'lm:Qwen/Qwen2.5-Omni-7B-GPTQ-Int4',
    }:
        # vLLM-Omni overrides these checkpoints' architecture with a speech-generation
        # model returning a tuple. The native vLLM thinker provides label logits.
        os.environ['VLLM_PLUGINS'] = ''
    state = {'spec': args.spec, 'pid': os.getpid(), 'state': 'loading', 'started_at': time.time()}

    def save(**kwargs):
        state.update(kwargs, updated_at=time.time())
        args.status.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(args.status)

    save()
    try:
        model = make_model(args.spec)
        if 'audio' not in model.modalities:
            raise RuntimeError('Adapter does not declare native audio support')
        tasks = [(t, dataio.load_task(t)) for t in tasks_under('aud')]
        if len(tasks) != 21 or sum(len(items) for _, items in tasks) != 10560:
            raise RuntimeError('Expected the frozen 21-task / 10,560-item suite')
        model.load()  # fail before creating thousands of failed calls
        save(state='smoke', model_id=model.model_id)
        task, items = tasks[0]
        runner.run_task(model, task.name, items[:2], k=1, workers=1)
        smoke_ids = {i.id for i in items[:2]}
        rows = [r for r in runner.latest_rows(model.slug, task.name) if r['item_id'] in smoke_ids]
        if len(rows) != 2 or any(r['response'].get('error') or not r['response'].get('answers') for r in rows):
            raise RuntimeError('Smoke gate failed; inspect the two cached rows, full evaluation was not started')
        save(state='running')
        for task, items in tasks:
            for offset in range(0, len(items), 32):
                batch = items[offset:offset + 32]
                prior = {r['item_id']: r for r in runner.latest_rows(model.slug, task.name)}
                repair = [i for i in batch if prior.get(i.id, {}).get('response', {}).get('error') ==
                          'rejected: this adapter answers one question per request']
                repair_ids = {i.id for i in repair}
                if repair:
                    runner.run_task(model, task.name, repair, k=1, workers=1, retry_rejected=True)
                runner.run_task(model, task.name, [i for i in batch if i.id not in repair_ids], k=1, workers=1)
                ids = {i.id for i in batch}
                rows = [r for r in runner.latest_rows(model.slug, task.name) if r['item_id'] in ids]
                technical = [r for r in rows if r['response'].get('error') and not r['response']['error'].startswith('rejected:')]
                save(task=task.name, task_processed=min(offset + 32, len(items)), task_total=len(items))
                if len(technical) >= 3:
                    raise RuntimeError(f'{len(technical)} technical errors in batch; stopped to avoid cascading failures')
        save(state='completed')
    except BaseException as exc:
        save(state='blocked', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if 'model' in locals():
            plan = sweep.Plan(name='audio_' + model.slug, tasks=['aud'], envs=[],
                              models=[sweep.ModelPlan(spec=args.spec, envs=False)])
            try:
                sweep.write_report(plan, sweep.report(plan))
            except Exception as exc:
                save(report_error=str(exc))
            finally:
                close = getattr(model, 'close', None)
                if callable(close):
                    close()  # release the vLLM engine so the queue can advance after success or failure



if __name__ == '__main__':
    main()
