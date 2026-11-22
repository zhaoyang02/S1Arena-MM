from jevarena_mm.models.jev_omni import JevOmni
from jevarena_mm.schema import Answer, DecisionRequest, ModelResponse


class HeadStub(JevOmni):
    def _decide(self, request):
        if len(request.questions) != 1:
            return super()._decide(request)
        name = next(iter(request.questions))
        assert request.state['clip'].sha256 == 'a' * 64
        return ModelResponse(answers={name: Answer(probs={'true': .8, 'false': .2})}, latency_ms=5)


def request(kind):
    return DecisionRequest(state={'clip': {'kind': kind, 'uri': 'toy/a.wav', 'sha256': 'a' * 64}},
                           questions={name: {'type': 'boolean', 'instructions': name} for name in ['q1', 'q2']})


def test_audio_head_answers_every_question_and_sums_latency():
    response = HeadStub().decide(request('audio'))
    assert response.error is None
    assert set(response.answers) == {'q1', 'q2'}
    assert response.latency_ms == 10
    assert response.usage['audio_question_protocol'] == 'independent-native-head-v1'


def test_video_behavior_is_unchanged():
    assert HeadStub().decide(request('video')).error == 'rejected: this adapter answers one question per request'
