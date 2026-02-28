const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:36300";

export interface TranscriptionResult {
    text: string;
    segments: { start: number; end: number; text: string }[];
    language: string;
    language_probability: number;
    duration: number;
    processing_time: number;
}

export interface Recording {
    id: string;
    filename: string;
    size_bytes: number;
    duration: number;
    created_at: string;
}

export interface ModelOption {
    engine: string;
    model: string;
    label: string;
    installed?: boolean;
}

export interface ModelSelection {
    engine: string;
    model: string;
}

export interface ModelOptionsResponse {
    options: Record<string, ModelOption[]>;
    current: Record<string, ModelSelection>;
    gpu_available: boolean;
}

export interface SystemInfo {
    hardware: {
        mode: string;
        tier: string;
        cpu_cores: number;
        ram_gb: number;
        gpu_available: boolean;
        gpu_name: string | null;
        vram_gb: number;
    };
    models: {
        configured: Record<string, { engine: string; model: string; device: string }>;
        loaded: string[];
    };
}

export interface EmotionScore {
    label: string;
    score: number;
}

export interface AudioFeatures {
    duration: number;
    pitch: {
        mean_hz: number | null;
        min_hz: number | null;
        max_hz: number | null;
        std_hz: number | null;
    };
    energy: {
        mean_db: number;
        max_db: number;
        dynamic_range_db: number;
    };
    tempo_bpm: number;
    spectral: {
        centroid_mean_hz: number;
        rolloff_mean_hz: number;
        zero_crossing_rate: number;
    };
    speaking_rate: {
        syllables_per_sec: number;
        estimated_wpm: number;
    };
    mfcc_means: number[];
}

export interface DiarizationSegment {
    speaker: string;
    start: number;
    end: number;
    duration: number;
}

export interface DiarizationResult {
    segments: DiarizationSegment[];
    summary: {
        num_speakers: number;
        speakers: Record<string, number>;
    };
    error?: string;
}

export interface AnalysisResult {
    features: AudioFeatures;
    emotions: EmotionScore[];
    primary_emotion?: string;
    emotion_error?: string;
    emotion_available: boolean;
    diarization: DiarizationResult;
    diarization_available: boolean;
    processing_time: number;
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
    let res: Response;
    try {
        res = await fetch(`${API_BASE}${path}`, init);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
            throw new Error("Connection to API failed. The server may be busy starting a Docker container (30-90s). Try again.");
        }
        throw new Error(`Network error: ${msg}`);
    }
    if (!res.ok) {
        const body = await res.text().catch(() => "");
        // Parse JSON error detail if possible
        try {
            const parsed = JSON.parse(body);
            if (parsed.detail) {
                throw new Error(`API ${res.status}: ${typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail)}`);
            }
        } catch { /* not JSON, fall through */ }
        throw new Error(`API ${res.status}: ${body}`);
    }
    return res;
}

export async function transcribe(file: Blob, language?: string): Promise<TranscriptionResult> {
    const form = new FormData();
    form.append("file", file, "recording.wav");
    if (language) form.append("language", language);
    const res = await apiFetch("/api/transcribe", { method: "POST", body: form });
    return res.json();
}

export async function synthesize(text: string, engine?: string, model?: string, voice?: string): Promise<Blob> {
    const body: Record<string, string> = { text };
    if (engine) body.engine = engine;
    if (model) body.model = model;
    if (voice) body.voice = voice;
    const res = await apiFetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    return res.blob();
}

export async function getRecordings(): Promise<Recording[]> {
    const res = await apiFetch("/api/recordings");
    const data = await res.json();
    return data.recordings;
}

export async function uploadRecording(file: Blob): Promise<Recording> {
    const form = new FormData();
    form.append("file", file, "recording.wav");
    const res = await apiFetch("/api/recordings", { method: "POST", body: form });
    return res.json();
}

export async function deleteRecording(id: string): Promise<void> {
    await apiFetch(`/api/recordings/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function getRecordingAudioUrl(id: string): string {
    return `${API_BASE}/api/recordings/${encodeURIComponent(id)}/audio`;
}

export async function getRecordingBlob(id: string): Promise<Blob> {
    const res = await apiFetch(`/api/recordings/${encodeURIComponent(id)}/audio`);
    return res.blob();
}

export async function getSystemInfo(): Promise<SystemInfo> {
    const res = await apiFetch("/api/system/info");
    return res.json();
}

export async function getModelOptions(): Promise<ModelOptionsResponse> {
    const res = await apiFetch("/api/system/model-options");
    return res.json();
}

export interface SwitchModelResult {
    status: string;
    category: string;
    engine: string;
    model: string;
    device: string;
    docker_status?: string;
}

export async function switchModel(category: string, engine: string, model: string): Promise<SwitchModelResult> {
    const res = await apiFetch("/api/system/switch-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, engine, model }),
    });
    return res.json();
}

export async function checkHealth(): Promise<boolean> {
    try {
        await apiFetch("/health");
        return true;
    } catch {
        return false;
    }
}

export async function analyze(file: Blob): Promise<AnalysisResult> {
    const form = new FormData();
    form.append("file", file, "recording.wav");
    const res = await apiFetch("/api/analyze", { method: "POST", body: form });
    return res.json();
}
