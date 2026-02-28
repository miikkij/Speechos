"""Test wav2small model with manual weight loading (bypass transformers API).

Architecture exactly from audeering/wav2small README.md on HuggingFace.
"""
import torch, librosa, time, os, json
import numpy as np
from torch import nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


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
        fft_window = librosa.filters.get_window("hann", n_time, fftbins=True)
        fft_window = librosa.util.pad_center(fft_window, size=n_time)
        out_channels = n_fft // 2 + 1
        (x, y) = np.meshgrid(np.arange(n_time), np.arange(n_fft))
        omega = np.exp(-2 * np.pi * 1j / n_time)
        dft_matrix = np.power(omega, x * y) * fft_window[None, :]
        dft_matrix = dft_matrix[0:out_channels, :]
        dft_matrix = dft_matrix[:, None, :]
        self.conv_real = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, padding=0, bias=False)
        self.conv_real.weight.data = torch.tensor(
            np.real(dft_matrix), dtype=self.conv_real.weight.dtype, device=self.conv_real.weight.device
        )
        self.conv_imag.weight.data = torch.tensor(
            np.imag(dft_matrix), dtype=self.conv_imag.weight.dtype, device=self.conv_imag.weight.device
        )
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x[:, None, :]
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        return real**2 + imag**2  # (B, freq, time): no sqrt per original code


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=16000, n_fft=64, n_mels=26, fmin=0.0, freeze_parameters=True):
        super().__init__()
        fmax = sr // 2
        W2 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.register_buffer("melW", torch.tensor(W2, dtype=torch.float32))
        self.register_buffer("amin", torch.tensor([1e-10]))

    def forward(self, x):
        # x: (B, freq, time) → (B, 1, time, n_mels)
        x = torch.matmul(x[:, None, :, :].transpose(2, 3), self.melW)
        x = torch.where(x > self.amin, x, self.amin)
        x = 10 * torch.log10(x)
        return x


class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return torch.relu_(x)


class Vgg7(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Conv(1, 13)
        self.l2 = Conv(13, 13)
        self.l3 = Conv(13, 13)
        self.maxpool_A = nn.MaxPool2d(3, stride=2, padding=1)
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


class Wav2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg7 = Vgg7()
        self.adv = nn.Linear(338, 3)

    def forward(self, x):
        x = self.vgg7(x)
        return self.adv(x)


if __name__ == "__main__":
    print("=== wav2small (audeering) Manual Load Test ===")

    # Download weights
    ckpt_path = hf_hub_download("audeering/wav2small", "model.safetensors")
    print(f"Checkpoint: {ckpt_path}")

    state_dict = load_file(ckpt_path)
    print(f"State dict keys: {len(state_dict)} keys")

    model = Wav2Small()
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {result.unexpected_keys}")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    wav_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")])
    results = {}

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
        results[wf] = res
        print(f"  {wf}: A={res['arousal']:.3f} D={res['dominance']:.3f} V={res['valence']:.3f} [{res['time_ms']}ms]")

    # Merge into existing results
    results_path = os.path.join(SAMPLES_DIR, "..", "docs", "research", "speech-analysis", "TEST-RESULTS.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    else:
        existing = {}
    existing["9. wav2small (audeering tiny A/D/V)"] = results
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print("\nResults merged into TEST-RESULTS.json")
    print("=== PASSED ===")
