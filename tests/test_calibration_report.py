from types import SimpleNamespace

import pytest

from jevarena_mm import metrics, sweep
from jevarena_mm.schema import DecisionRequest, Gold, Item, Question


def test_sweep_retains_actual_calibration_and_pending(monkeypatch):
    task = SimpleNamespace(name='aud.basic/toy/source', leaf='aud.basic/toy', status='preview')
    items = [Item(leaf=task.leaf, task=task.name,
                  request=DecisionRequest(state={'n': i}, questions={'q': Question(type='boolean', instructions='Yes?')}),
                  gold={'q': Gold(label='true')}) for i in range(3)]
    rows = [{'item_id': items[0].id, 'sample': 0, 'response': {
                'answers': {'q': {'probs': {'true': .75, 'false': .25}}}, 'latency_ms': 10}},
            {'item_id': items[1].id, 'sample': 0, 'response': {'answers': {}, 'error': 'failed'}}]
    model = SimpleNamespace(model_id='toy', slug='toy')
    monkeypatch.setattr(sweep, 'plan_tasks', lambda _: [task])
    monkeypatch.setattr(sweep, 'plan_envs', lambda _: [])
    monkeypatch.setattr(sweep, 'make_model', lambda _: model)
    monkeypatch.setattr(sweep.dataio, 'load_task', lambda _: items)
    monkeypatch.setattr(sweep.runner, 'latest_rows', lambda *_: rows)
    rep = sweep.report(sweep.Plan(name='toy', models=[sweep.ModelPlan(spec='toy')]))
    cell = rep['task_metrics'][task.name]['toy']
    expected = metrics.task_metrics(task, items[:2], rows)
    assert cell['brier'] == pytest.approx((.125 + .5) / 2)
    assert cell['ece'] == expected['ece']
    assert cell['unanswered'] == 1 and cell['pending_items'] == 1
    assert rep['static'] == {} and rep['overall']['toy']['static_accuracy'] is None
    assert '0.312' in sweep.report_markdown(rep)



def test_video_report_has_no_added_calibration_section():
    rep = {"plan": "video", "models": [], "overall": {}, "static": {}, "preview": {}, "envs": {},
           "task_metrics": {"vid.basic/toy/source": {"m": {"brier": .25}}}}
    assert "Calibration and coverage" not in sweep.report_markdown(rep)
