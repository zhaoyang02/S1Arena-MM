import importlib.util
import json
from pathlib import Path

import pytest

from jevarena_mm.schema import Answer, ModelResponse
from jevarena_mm.streaming.resume import reuse


def test_resume_keeps_first_success_and_retries_failures(tmp_path):
    progress = {'protocol': 'p', 'manifest_sha256': 'h', 'model': 'm', 'backend': 'qwen',
                'config': {'x': 1}, 'case_ids': ['a', 'b']}
    sources = []
    for index in range(2):
        source = tmp_path / str(index)
        source.mkdir()
        rows = [{'id': 'a', 'status': 'completed', 'score': index}, {'id': 'b', 'status': 'failed'}]
        (source / 'progress.json').write_text(json.dumps({**progress, 'results': rows}))
        (source / 'a').mkdir()
        (source / 'a/events.jsonl').write_text('{"event":"observation_end"}\n')
        sources.append(source)
    out = tmp_path / 'out'
    out.mkdir()
    rows = reuse(sources, progress, out)
    assert len(rows) == 1 and rows[0]['score'] == 0
    assert progress['analysis']['new_runtime_case_ids'] == ['b']
    assert (out / 'a/events.jsonl').exists()
    assert not (out / 'b').exists()


@pytest.mark.parametrize('field', ['manifest_sha256', 'model', 'config'])
def test_resume_refuses_different_experiment(tmp_path, field):
    old = {'manifest_sha256': 'h', 'model': 'm', 'config': {}, 'results': []}
    (tmp_path / 'progress.json').write_text(json.dumps(old))
    with pytest.raises(ValueError, match='identity mismatch'):
        reuse([tmp_path], {**old, field: 'different', 'case_ids': []}, tmp_path)


def test_controller_request_does_not_leak_gold_or_transcript(tmp_path):
    path = Path(__file__).parents[1] / 'scripts/audio/controller_server.py'
    spec = importlib.util.spec_from_file_location('controller_server_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / 'observed.wav'
    audio.write_bytes(b'observed-only')
    incoming = {'speaking': False, 'key': 'case:1', 'gold': 'speak', 'transcript': 'SECRET', 'future_audio': 'SECRET'}
    actions, request = module.make_request(incoming, audio)
    assert set(actions) == {'wait', 'speak'}
    assert 'SECRET' not in request.model_dump_json()
    assert set(request.state) == {'state', 'audio'}
    assert module.make_request(incoming, audio)[0] == actions
    response = ModelResponse(answers={'action': Answer(probs={'wait': 0.2, 'speak': 0.8})})
    assert module.choose(response, actions)[0] == 'speak'
    with pytest.raises(ValueError):
        module.choose(ModelResponse(error='failed'), actions)
    response.answers['action'].probs['speak'] = float('nan')
    with pytest.raises(ValueError):
        module.choose(response, actions)
