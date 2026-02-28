"""Fix and re-run the 3 failed tests from test_all_models.py"""
import json, time, os, traceback
import numpy as np
import librosa
import torch

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
wav_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")])
results = {}
all_errors = []


def run_section(name, func):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    try:
        result = func()
        results[name] = result
        print(f"  STATUS: OK")
        return True
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        results[name] = {"error": error_msg}
        all_errors.append((name, error_msg))
        print(f"  STATUS: FAILED - {error_msg}")
        traceback.print_exc()
        return False


# ============================================================
# Fix 1: Dpngtm: correct model name
# ============================================================
def test_wav2vec2_dpngtm():
    from transformers import pipeline

    print("    Loading Dpngtm/wav2vec2-emotion-recognition...")
    classifier = pipeline(
        "audio-classification",
        model="Dpngtm/wav2vec2-emotion-recognition",
        device=0 if torch.cuda.is_available() else -1,
    )

    w2v_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        preds = classifier({"raw": y, "sampling_rate": 16000}, top_k=None)
        elapsed = time.time() - t0

        top = preds[0]
        all_scores = {p["label"]: round(p["score"], 4) for p in preds}
        res = {
            "top_emotion": top["label"],
            "top_score": round(top["score"], 4),
            "all_scores": all_scores,
            "time_ms": round(elapsed * 1000, 1),
        }
        w2v_results[wf] = res
        print(f"    {wf}: {res['top_emotion']} ({res['top_score']:.2%}) [{res['time_ms']}ms]")
    return w2v_results


# ============================================================
# Fix 2: HuBERT SER
# ============================================================
def test_hubert_ser():
    from transformers import pipeline

    print("    Loading xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned...")
    # Using a HuBERT-based SER model
    classifier = pipeline(
        "audio-classification",
        model="superb/hubert-large-superb-er",
        device=0 if torch.cuda.is_available() else -1,
    )

    hubert_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        preds = classifier({"raw": y, "sampling_rate": 16000}, top_k=None)
        elapsed = time.time() - t0

        top = preds[0]
        all_scores = {p["label"]: round(p["score"], 4) for p in preds}
        res = {
            "top_emotion": top["label"],
            "top_score": round(top["score"], 4),
            "all_scores": all_scores,
            "time_ms": round(elapsed * 1000, 1),
        }
        hubert_results[wf] = res
        print(f"    {wf}: {res['top_emotion']} ({res['top_score']:.2%}) [{res['time_ms']}ms]")
    return hubert_results


# ============================================================
# Fix 3: wav2small: use custom model class per audeering docs
# Note: Superseded by manual loading in test_all_models.py::test_wav2small
# ============================================================
def _wav2small_legacy():
    from transformers import Wav2Vec2PreTrainedModel, PretrainedConfig
    from torch import nn

    print("    Loading audeering/wav2small with custom model class...")

    def _prenorm(x, attention_mask=None):
        if attention_mask is not None:
            N = attention_mask.sum(1, keepdim=True)
            x -= x.sum(1, keepdim=True) / N
            var = (x * x).sum(1, keepdim=True) / N
        else:
            x -= x.mean(1, keepdim=True)
            var = (x * x).mean(1, keepdim=True)
        return x / torch.sqrt(var + 1e-7)

    class Spectrogram(nn.Module):
        def __init__(self, n_fft=64, n_time=64, hop_length=32, freeze_parameters=True):
            super().__init__()
            fft_window = librosa.filters.get_window('hann', n_time, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, size=n_time)
            out_channels = n_fft // 2 + 1
            (x, y) = np.meshgrid(np.arange(n_time), np.arange(n_fft))
            omega = np.exp(-2 * np.pi * 1j / n_time)
            dft_matrix = np.power(omega, x * y)
            dft_matrix = dft_matrix * fft_window[None, :]
            dft_matrix = dft_matrix[0:out_channels, :]
            dft_matrix = dft_matrix[:, None, :]
            self.conv_real = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, padding=0, bias=False)
            self.conv_imag = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, padding=0, bias=False)
            self.conv_real.weight.data.copy_(torch.tensor(np.real(dft_matrix)[:, :, :n_fft], dtype=torch.float32))
            self.conv_imag.weight.data.copy_(torch.tensor(np.imag(dft_matrix)[:, :, :n_fft], dtype=torch.float32))
            if freeze_parameters:
                for param in self.parameters():
                    param.requires_grad = False

        def forward(self, x):
            x = x[:, None, :]
            real = self.conv_real(x)
            imag = self.conv_imag(x)
            return (real ** 2 + imag ** 2) ** 0.5

    class LogmelFilterBank(nn.Module):
        def __init__(self, sr=16000, n_fft=64, n_mels=26, freeze_parameters=True):
            super().__init__()
            mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
            self.mel = nn.Parameter(torch.tensor(mel_fb, dtype=torch.float32), requires_grad=not freeze_parameters)

        def forward(self, x):
            x = torch.matmul(self.mel, x)
            return torch.log(x + 1e-7)

    class Conv(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.conv = nn.Conv2d(ci, co, 3, padding=1, stride=1)
            self.bn = nn.BatchNorm2d(co)

        def forward(self, x):
            return self.bn(self.conv(x)).relu()

    class Vgg7(nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool_A = nn.MaxPool2d(kernel_size=(2, 1))
            self.l1 = Conv(1, 13)
            self.l2 = Conv(13, 13)
            self.l3 = Conv(13, 13)
            self.l4 = Conv(13, 13)
            self.l5 = Conv(13, 13)
            self.l6 = Conv(13, 13)
            self.l7 = Conv(13, 13)
            self.lin = nn.Conv2d(13, 13, 1, padding=0, stride=1)
            self.sof = nn.Conv2d(13, 13, 1, padding=0, stride=1)
            self.spectrogram_extractor = Spectrogram()
            self.logmel_extractor = LogmelFilterBank()

        def forward(self, x, attention_mask=None):
            x = _prenorm(x, attention_mask=attention_mask)
            x = self.spectrogram_extractor(x)
            x = self.logmel_extractor(x)
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            x = self.maxpool_A(x)
            x = self.l4(x)
            x = self.l5(x)
            x = self.l6(x)
            x = self.l7(x)
            x = self.lin(x) * self.sof(x).softmax(2)
            x = x.sum(2)
            x = torch.cat([x, torch.bmm(x, x.transpose(1, 2))], 2)
            return x.reshape(-1, 338)

    class Wav2SmallConfig(PretrainedConfig):
        model_type = "wav2vec2"
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.half_mel = 13
            self.n_fft = 64
            self.n_time = 64
            self.hidden = 2 * self.half_mel * self.half_mel
            self.hop = self.n_time // 2

    class Wav2Small(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.vgg7 = Vgg7()
            self.adv = nn.Linear(config.hidden, 3)

        def forward(self, x, attention_mask=None):
            x = self.vgg7(x, attention_mask=attention_mask)
            return self.adv(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Small.from_pretrained("audeering/wav2small").to(device).eval()

    w2s_results = {}
    for wf in wav_files:
        path = os.path.join(SAMPLES_DIR, wf)
        t0 = time.time()
        y, sr = librosa.load(path, sr=16000)
        signal = torch.from_numpy(y)[None, :].to(device)
        with torch.no_grad():
            logits = model(signal)
        elapsed = time.time() - t0

        res = {
            "arousal": round(float(logits[0, 0]), 4),
            "dominance": round(float(logits[0, 1]), 4),
            "valence": round(float(logits[0, 2]), 4),
            "time_ms": round(elapsed * 1000, 1),
        }
        w2s_results[wf] = res
        print(f"    {wf}: A={res['arousal']:.3f} D={res['dominance']:.3f} V={res['valence']:.3f} [{res['time_ms']}ms]")
    return w2s_results


if __name__ == "__main__":
    run_section("7. wav2vec2 SER (Dpngtm)", test_wav2vec2_dpngtm)
    run_section("8. HuBERT SER (superb-er)", test_hubert_ser)
    run_section("9. wav2small (audeering tiny A/D/V)", _wav2small_legacy)

    print("\n" + "=" * 80)
    print("RE-RUN SUMMARY")
    print("=" * 80)
    passed = [k for k, v in results.items() if "error" not in v]
    failed = [k for k, v in results.items() if "error" in v]
    print(f"  Passed: {len(passed)}/{len(results)}")
    if failed:
        for name, err in all_errors:
            print(f"  FAILED: {name}: {err}")

    # Merge into existing results
    existing_path = os.path.join(SAMPLES_DIR, "..", "docs", "research", "speech-analysis", "TEST-RESULTS.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        existing.update(results)
        with open(existing_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"  Results merged into {existing_path}")
    else:
        with open(existing_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
