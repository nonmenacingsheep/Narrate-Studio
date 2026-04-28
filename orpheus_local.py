"""
Local Orpheus TTS inference using transformers + SNAC.
Bypasses vLLM (Linux-only) and the orpheus_tts package entirely
to avoid the vllm._C import error on Windows.
"""
import os
import torch
import numpy as np
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODEL_ID  = "canopylabs/orpheus-tts-0.1-finetune-prod"
SNAC_ID   = "hubertsiuzdak/snac_24khz"
_START    = 128259
_END      = [128009, 128260, 128261, 128257]
_STOP     = 49158


def _load_hf_token():
    path = os.path.join(os.path.dirname(__file__), "hf_token.txt")
    if os.path.exists(path):
        t = open(path).read().strip()
        return t or None
    return None


# ── Stop generation as soon as the first non-audio token appears after audio ──

class _AudioEndCriteria(StoppingCriteria):
    """Halt model.generate() on the first non-audio token after audio has begun.
    Audio tokens all have IDs >= 128266 (audio_base 128256 + 10)."""
    _AUDIO_START = 128266

    def __init__(self):
        self._seen_audio = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last = input_ids[0, -1].item()
        if last >= self._AUDIO_START:
            self._seen_audio = True
        elif self._seen_audio:
            return True   # non-audio token after audio → end of speech
        return False


# ── SNAC decoder (inlined from orpheus_tts.decoder to avoid vLLM import) ──

_snac_model = None

def _get_snac(device):
    global _snac_model
    if _snac_model is None:
        _snac_model = SNAC.from_pretrained(SNAC_ID).eval().to(device)
    return _snac_model

def _convert_to_audio(multiframe, device):
    if len(multiframe) < 7:
        return None
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    c0, c1, c2 = [], [], []
    for j in range(num_frames):
        i = j * 7
        c0.append(frame[i])
        c1 += [frame[i+1], frame[i+4]]
        c2 += [frame[i+2], frame[i+3], frame[i+5], frame[i+6]]

    def t(lst):
        return torch.tensor(lst, device=device, dtype=torch.int32).unsqueeze(0)

    codes = [t(c0), t(c1), t(c2)]
    if any(torch.any(c < 0) or torch.any(c > 4096) for c in codes):
        return None

    snac = _get_snac(device)
    with torch.inference_mode():
        audio = snac.decode(codes)

    audio_slice = audio[:, :, 2048:4096]
    pcm = (audio_slice.detach().cpu().numpy() * 32767).astype(np.int16)
    return pcm.tobytes()


# ── Main model class ────────────────────────────────────────────────────────

class OrpheusLocal:
    def __init__(self, device="cuda"):
        self.device = device
        token = _load_hf_token()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            token=token,
        ).to(device)
        self.model.eval()

        # Resolve the vocab ID for audio tokens.
        # <custom_token_10> is the first SNAC code token.
        ct10 = self.tokenizer.convert_tokens_to_ids("<custom_token_10>")
        if ct10 == self.tokenizer.unk_token_id or ct10 is None:
            ids = self.tokenizer.encode("<custom_token_10>", add_special_tokens=False)
            ct10 = ids[0] if ids else None
        self._audio_base = (ct10 - 10) if ct10 is not None else None
        print(f"[Orpheus] <custom_token_10> vocab ID: {ct10}  |  audio_base: {self._audio_base}")

    def _build_input_ids(self, text: str, voice: str):
        """
        Replicates engine_class._format_prompt exactly:
        build raw IDs → decode to string → re-encode.
        The roundtrip is necessary — vLLM uses the string form and
        the tokenizer maps special tokens differently on re-encode.
        """
        adapted      = f"{voice}: {text}"
        prompt_ids   = self.tokenizer(adapted, return_tensors="pt").input_ids
        start        = torch.tensor([[_START]], dtype=torch.int64)
        end          = torch.tensor([_END],    dtype=torch.int64)
        raw_ids      = torch.cat([start, prompt_ids, end], dim=1)

        # decode → re-encode roundtrip
        prompt_str   = self.tokenizer.decode(raw_ids[0], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
        final        = self.tokenizer(prompt_str, return_tensors="pt",
                                      add_special_tokens=False)
        ids          = final.input_ids.to(self.device)
        mask         = torch.ones_like(ids)
        return ids, mask

    def generate_speech(self, text: str, voice: str = "tara",
                        temperature: float = 0.6, top_p: float = 0.8,
                        max_new_tokens: int = 1800,
                        repetition_penalty: float = 1.3) -> np.ndarray:
        ids, mask = self._build_input_ids(text, voice)

        with torch.inference_mode():
            out = self.model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=_STOP,
                pad_token_id=_STOP,
                stopping_criteria=StoppingCriteriaList([_AudioEndCriteria()]),
            )

        generated = out[0][ids.shape[1]:].tolist()
        return self._tokens_to_audio(generated)

    def _snac_code(self, token_id: int, index: int):
        if self._audio_base is None:
            return None
        x    = token_id - self._audio_base   # == number in <custom_token_X>
        code = x - 10 - ((index % 7) * 4096)
        return code if 0 <= code <= 4096 else None

    def _tokens_to_audio(self, token_ids: list) -> np.ndarray:
        buffer, count, chunks = [], 0, []
        for tid in token_ids:
            code = self._snac_code(tid, count)
            if code is None:
                if count > 0:
                    break   # non-audio token after audio started → end of speech
                continue
            buffer.append(code)
            count += 1
            if count % 7 == 0 and count >= 28:
                raw = _convert_to_audio(buffer[-28:], self.device)
                if raw is not None:
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    chunks.append(pcm)

        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
