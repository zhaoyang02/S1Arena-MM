"""Loopback decision-only service for the separately labelled shared-voice track."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jevarena_mm.registry import make_model
from jevarena_mm.schema import DecisionRequest, MediaRef, Question

DESCRIPTIONS = {
    'continue': 'Continue the current response because the user only acknowledges it.',
    'stop': 'Stop the current response and address the new user request.',
    'wait': 'Keep listening because the user has not finished their turn.',
    'speak': 'Start a response because the user has finished their turn.',
}


def make_request(req, path):
    """Only arrived audio and current playback state; no transcript, future audio or gold."""
    if type(req['speaking']) is not bool:
        raise ValueError('speaking must be boolean')
    actions = ['continue', 'stop'] if req['speaking'] else ['wait', 'speak']
    if int(hashlib.sha256(req['key'].encode()).hexdigest(), 16) % 2:
        actions.reverse()
    state = 'You are currently speaking.' if req['speaking'] else 'You are listening and not currently speaking.'
    state += ' The audio contains only user audio observed so far in this session. Decide what to do at the latest detected pause.'
    ref = MediaRef(kind='audio', uri=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return actions, DecisionRequest(state={'state': state, 'audio': ref}, questions={
        'action': Question(type='choice', instructions='Which action should the assistant take now?',
                           criteria={a: DESCRIPTIONS[a] for a in actions})})


def choose(response, actions):
    if response.error or response.status != 'ok' or 'action' not in response.answers:
        raise ValueError('Controller did not return a valid decision')
    probs = response.answers['action'].probs
    if set(probs) != set(actions) or any(not math.isfinite(v) or v < 0 for v in probs.values()) or abs(sum(probs.values())-1) > 1e-4:
        raise ValueError('Invalid controller probability distribution')
    action = max(actions, key=lambda a: probs[a])
    return action, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    if args.spec.split('@', 1)[0] in {'lm:Qwen/Qwen2.5-Omni-7B-AWQ', 'lm:Qwen/Qwen2.5-Omni-7B-GPTQ-Int4'}:
        os.environ['VLLM_PLUGINS'] = ''
    with tempfile.TemporaryDirectory(prefix='audio-controller-') as temp:
        os.environ['JEVARENA_MM_MEDIA'] = temp
        model = make_model(args.spec)
        if 'audio' not in model.modalities:
            raise ValueError('Controller must accept audio')
        model.load()
        identity = {'ready': True, 'protocol': 'audio-shared-voice-controller-v1',
                    'model_id': model.model_id, 'model_spec': args.spec,
                    'server_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'adapter_sha256': hashlib.sha256(Path(sys.modules[type(model).__module__].__file__).read_bytes()).hexdigest(),
                    'voice_revision': '26291f793822fb6be9555850f06dfe95f2d7e695',
                    'vad_threshold': 0.5, 'silence_ms': 500, 'padding_ms': 300,
                    'option_order': 'sha256(case_id:decision_number), label-independent'}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(identity).encode())

            def do_POST(self):
                try:
                    size = int(self.headers['Content-Length'])
                    if not 0 < size < 6000000:
                        raise ValueError('Invalid input size')
                    req = json.loads(self.rfile.read(size))
                    pcm = base64.b64decode(req['pcm'], validate=True)
                    if not 0 < len(pcm) <= 16000 * 2 * 120 or len(pcm) % 2:
                        raise ValueError('Invalid PCM')
                    with tempfile.NamedTemporaryFile(dir=temp, suffix='.wav') as f:
                        path = Path(f.name)
                        with wave.open(str(path), 'wb') as wav:
                            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                            wav.writeframes(pcm)
                        actions, request = make_request(req, path)
                        start = time.monotonic()
                        response = model.decide(request)
                        action, probs = choose(response, actions)
                    result = {'action': action, 'option_order': actions, 'prediction_index': actions.index(action),
                              'confidence': probs[action], 'probs': probs, 'inference_s': time.monotonic()-start,
                              'input_samples': len(pcm)//2, 'model_id': model.model_id}
                    self.send_response(200)
                except Exception as exc:
                    result = {'error': type(exc).__name__}
                    print(f'Controller error: {type(exc).__name__}: {exc}', flush=True)
                    self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
        try:
            print(json.dumps(identity), flush=True)
            HTTPServer(('127.0.0.1', 8767), Handler).serve_forever()
        finally:
            close = getattr(model, 'close', None)
            if callable(close):
                close()


if __name__ == '__main__':
    main()
