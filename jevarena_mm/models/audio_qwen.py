"""Pinned Qwen3-Omni native audio, local CUDA inference with verbalized probabilities."""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time

from .. import render
from .audio_common import audio_ref, waveform, reparse
from ..registry import register_model
from ..schema import ModelResponse
from .base import ModelAdapter

MODEL = 'Qwen/Qwen3-Omni-30B-A3B-Instruct'
REVISION = '26291f793822fb6be9555850f06dfe95f2d7e695'


@register_model
class AudioQwen(ModelAdapter):
    kind = 'audio_qwen'
    modalities = frozenset({"audio"})
    native_probs = False
    probability_source = 'verbalized'

    def __init__(self, model_id=MODEL):
        if model_id != MODEL:
            raise ValueError('this adapter is validated for Qwen3-Omni-30B-A3B-Instruct only')
        self._lock = threading.Lock()
        self._model = self._processor = None

    @property
    def model_id(self):
        return f'audio-qwen/{MODEL}/{REVISION}/mm-v1-bf16-sdpa-t1-p1-k0-max4000-ctx16384@sample_rate=16000,mono=1,decoder=pyav,chunking=none'

    @staticmethod
    def conversation(request):
        audio_ref(request)
        messages = render.messages(request)
        # Audio is passed as an array to the processor; no path, provenance or gold in text.
        return [{'role': 'system', 'content': [{'type': 'text', 'text': messages[0]['content']}]},
                {'role': 'user', 'content': [{'type': 'audio', 'audio': 'attached'},
                                           {'type': 'text', 'text': messages[1]['content']}]}]

    def prepare(self):
        """Load once before a sweep; configuration failures abort rather than creating hundreds of errors."""
        with self._lock:
            self._load()

    def _load(self):
        if self._model is not None:
            return
        import torch
        import transformers
        from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        if transformers.__version__ != '4.57.1':
            raise RuntimeError('use pinned transformers==4.57.1 for this adapter version')
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available():
            raise RuntimeError('set CUDA_VISIBLE_DEVICES to allocated GPUs; CUDA is required')
        memory = {}
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            memory[index] = min(free - 4*1024**3, 42*1024**3)
            if memory[index] <= 0:
                raise RuntimeError('insufficient free GPU memory')
        memory['cpu'] = 0
        config = Qwen3OmniMoeConfig.from_pretrained(MODEL, revision=REVISION)
        config.enable_audio_output = False
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL, revision=REVISION)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, config=config, dtype=torch.bfloat16,
            device_map='auto', max_memory=memory, attn_implementation='sdpa',
            low_cpu_mem_usage=True).eval()
        if any(str(v) in ('cpu', 'disk') for v in model.hf_device_map.values()):
            raise RuntimeError('CPU/disk offload detected; allocate more GPU memory')
        self._model, self._processor = model, processor

    def _generate(self, request, audio, sample):
        import torch
        conversation = self.conversation(request)
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self._processor(text=prompt, audio=[audio], return_tensors='pt',
                                 audio_kwargs={'sampling_rate':16000, 'truncation':False},
                                 text_kwargs={'padding':True, 'truncation':False})
        # Whisper feature extraction uses 160 samples per frame; reject unexpected clipping.
        frames = int(inputs['feature_attention_mask'].sum().item())
        if abs(frames - len(audio)//160) > 2:
            raise ValueError('audio processor length mismatch; refusing truncated audio')
        length = inputs['input_ids'].shape[1]
        if length + 4000 > 16384:
            raise ValueError('context limit exceeded; no truncation performed')
        device = self._model.thinker.get_input_embeddings().weight.device
        inputs = inputs.to(device).to(torch.bfloat16)
        seed_data = json.dumps({'request': request.model_dump(mode='json'),
                                'sample':sample}, sort_keys=True)
        seed = int(hashlib.sha256(seed_data.encode()).hexdigest()[:8],16)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed)
            with torch.inference_mode():
                tokens, _ = self._model.generate(**inputs, return_audio=False,
                    thinker_do_sample=True, thinker_temperature=1.0, thinker_top_p=1.0,
                    thinker_top_k=0, thinker_max_new_tokens=4000)
        generated = tokens[:, length:]
        text = self._processor.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        return text, {'input_tokens':length,'output_tokens':generated.shape[1],
                      'audio_frames':frames,'seed':seed,'audio_preprocessing':'pyav_mono_16khz',
                      'output_limit_reached':generated.shape[1] >= 4000}

    def decide(self, request, sample=0):
        start = time.perf_counter()
        try:
            self.conversation(request)
            audio, input_usage = waveform(request)
        except (ValueError, OSError, KeyError) as error:
            return ModelResponse(error=f'rejected: {error}')
        with self._lock:
            self._load()
            try:
                text, usage = self._generate(request, audio, sample)
                usage['input'] = input_usage
                response = self.reparse(ModelResponse(raw_text=text, usage=usage), request)
            except ValueError as error:
                response = ModelResponse(error=f'rejected: {error}')
            except Exception as error:
                # Do not copy request data or arbitrary exception text into the cache.
                response = ModelResponse(error=f'local inference: {type(error).__name__}')
        return response.model_copy(update={'latency_ms':(time.perf_counter()-start)*1000,'cost_usd':None})

    def reparse(self, response, request):
        return reparse(response, request)
