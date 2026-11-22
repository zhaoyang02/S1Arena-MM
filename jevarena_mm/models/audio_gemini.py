"""Native WAV input through Gemini generateContent; verbalized probabilities.

Model availability is selected explicitly by the caller. No speech-to-text proxy.
API contract: https://ai.google.dev/api/generate-content
"""
from __future__ import annotations
import base64
import json
import os
import re
import time
import wave
from urllib.parse import urlparse

from .. import render
from .audio_common import waveform, wav_bytes, reparse
from ..env import load_env
from ..registry import register_model
from ..schema import ModelResponse
from .base import ModelAdapter, split_options

MAX_INLINE_BYTES = 20_000_000


@register_model
class AudioGemini(ModelAdapter):
    kind = 'audio_gemini'
    modalities = frozenset({"audio"})
    native_probs = False
    probability_source = 'verbalized'

    def __init__(self, model_id: str):
        model_id, options = split_options(model_id)
        if set(options) - {'sample_rate'}:
            raise ValueError('unknown audio_gemini option')
        self.sample_rate = int(options.get('sample_rate', 16000))
        if not 8000 <= self.sample_rate <= 48000:
            raise ValueError('sample_rate must be between 8000 and 48000')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', model_id):
            raise ValueError('provide a Gemini model ID, without models/ prefix')
        self._model_id = model_id

    @property
    def model_id(self):
        # Isolate cache from text channels and any future prompt/config revisions.
        return f'audio-gemini/{self._model_id}@sample_rate={self.sample_rate},mono=1,pcm=16,decoder=pyav,chunking=none,prompt=mm-v1,t=1,max_tokens=4000'

    def payload(self, request, media_part=None):
        if media_part is None:
            samples, _ = waveform(request, self.sample_rate)
            data = wav_bytes(samples, self.sample_rate)
            media_part = {'inlineData': {'mimeType': 'audio/wav', 'data': base64.b64encode(data).decode('ascii')}}
        messages = render.messages(request)
        body = {
            'systemInstruction': {'parts': [{'text': messages[0]['content']}]},
            'contents': [{'role': 'user', 'parts': [
                media_part,
                {'text': messages[1]['content']},
            ]}],
            'generationConfig': {'temperature': 1.0, 'maxOutputTokens': 4000,
                                 'responseMimeType': 'application/json'},
        }
        if len(json.dumps(body, ensure_ascii=False).encode()) > MAX_INLINE_BYTES:
            raise ValueError('request exceeds inline limit; no truncation performed')
        return body

    @staticmethod
    def _checked(result):
        if result.status_code >= 400:
            raise RuntimeError(f'HTTP {result.status_code}')
        return result

    def _upload(self, data, key, uploaded):
        import httpx

        base = 'https://generativelanguage.googleapis.com'
        start = self._checked(httpx.post(base + '/upload/v1beta/files', headers={
            'x-goog-api-key': key, 'X-Goog-Upload-Protocol': 'resumable',
            'X-Goog-Upload-Command': 'start',
            'X-Goog-Upload-Header-Content-Length': str(len(data)),
            'X-Goog-Upload-Header-Content-Type': 'audio/wav',
        }, json={'file': {'display_name': 'audio-input'}}, timeout=120.0))
        upload_url = start.headers['x-goog-upload-url']
        parsed = urlparse(upload_url)
        if parsed.scheme != 'https' or parsed.netloc != 'generativelanguage.googleapis.com':
            raise RuntimeError('unexpected upload origin')
        result = self._checked(httpx.post(upload_url, headers={
            'Content-Length': str(len(data)), 'X-Goog-Upload-Offset': '0',
            'X-Goog-Upload-Command': 'upload, finalize',
        }, content=data, timeout=120.0)).json()['file']
        name = result['name']
        if not re.fullmatch(r'files/[A-Za-z0-9_-]+', name):
            raise RuntimeError('invalid uploaded file name')
        uploaded.append(name)
        for _ in range(30):
            state = result.get('state')
            if state == 'ACTIVE':
                return {'fileData': {'mimeType': 'audio/wav', 'fileUri': result['uri']}}
            if state != 'PROCESSING':
                raise RuntimeError('uploaded audio is not active')
            time.sleep(1)
            result = self._checked(httpx.get(base + '/v1beta/' + name,
                                  headers={'x-goog-api-key': key}, timeout=30.0)).json()
        raise RuntimeError('audio processing timed out')

    def decide(self, request, sample=0):
        import httpx

        load_env()
        start = time.perf_counter()
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            return ModelResponse(error='configuration: GEMINI_API_KEY is missing')
        try:
            samples, input_usage = waveform(request, self.sample_rate)
            data = wav_bytes(samples, self.sample_rate)
            # Include all prompt bytes when choosing inline versus file transport.
            inline = {'inlineData': {'mimeType': 'audio/wav', 'data': base64.b64encode(data).decode('ascii')}}
            try:
                body = self.payload(request, inline)
            except ValueError as error:
                if 'inline limit' not in str(error):
                    raise
                body = None
        except (ValueError, KeyError, OSError, EOFError, wave.Error) as error:
            return ModelResponse(error=f'rejected: {type(error).__name__}: {error}')
        uploaded = []
        cleanup_failed = False
        try:
            if body is None:
                body = self.payload(request, self._upload(data, key, uploaded))
            result = self._checked(httpx.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{self._model_id}:generateContent',
                headers={'x-goog-api-key': key}, json=body, timeout=120.0,
            ))
            out = result.json()
            candidates = out.get('candidates') or []
            parts = candidates[0].get('content', {}).get('parts', []) if candidates else []
            text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
            response = self.reparse(ModelResponse(raw_text=text,
                usage=out.get('usageMetadata'), cost_usd=None), request)
        except Exception as error:
            # Do not cache provider error bodies, upload URLs, or request data.
            message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            response = ModelResponse(error=f'transport: {message}')
        finally:
            for name in uploaded:
                try:
                    deleted = httpx.delete('https://generativelanguage.googleapis.com/v1beta/' + name,
                                           headers={'x-goog-api-key': key}, timeout=30.0)
                    if deleted.status_code not in (200, 204, 404):
                        cleanup_failed = True
                except Exception:
                    cleanup_failed = True
        usage = dict(response.usage or {})
        usage['input'] = input_usage
        if cleanup_failed:
            usage['audio_upload_cleanup_failed'] = True
        response = response.model_copy(update={'usage': usage or None,
            'latency_ms': (time.perf_counter() - start) * 1000})
        return response

    def reparse(self, response, request):
        return reparse(response, request)
