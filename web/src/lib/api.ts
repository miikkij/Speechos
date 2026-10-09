const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:36300";

/** Language sent to the API. "auto" lets STT detect it and TTS use the configured voice. */
export type Language = "auto" | "fi" | "en";

export const LANGUAGES: { value: Language; label: string }[] = [
    { value: "fi", label: "Suomi" },
    { value: "en", label: "English" },
    { value: "auto", label: "Auto" },
];

/** File name with an extension that matches the blob's real container. */
function audioFileName(file: Blob): string {
    if (file instanceof File && file.name) return file.name;
    const type = file.type.toLowerCase();
    if (type.includes("webm")) return "recording.webm";
    if (type.includes("ogg")) return "recording.ogg";
    if (type.includes("mp4") || type.includes("aac")) return "recording.m4a";
    if (type.includes("mpeg")) return "recording.mp3";
    if (type.includes("flac")) return "recording.flac";
    return "recording.wav";
}

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

export interface SpeechActivity {
    segments: { start: number; end: number }[];
    speech_seconds: number;
    speech_ratio: number;
}

export interface AnalysisResult {
    features: AudioFeatures;
    emotions: EmotionScore[];
    primary_emotion?: string;
    emotion_error?: string;
    emotion_available: boolean;
    diarization: DiarizationResult;
    diarization_available: boolean;
    speech?: SpeechActivity | null;
    vad_available?: boolean;
    processing_time: number;
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
    const res = await fetch(`${API_BASE}${path}`, init);
    if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`API ${res.status}: ${body}`);
    }
    return res;
}

export async function transcribe(file: Blob, language: Language = "auto"): Promise<TranscriptionResult> {
    const form = new FormData();
    form.append("file", file, audioFileName(file));
    form.append("language", language);
    const res = await apiFetch("/api/transcribe", { method: "POST", body: form });
    return res.json();
}

export interface SynthesizeOptions {
    language?: Language;
    speed?: number;
    voice?: string;
}

export async function synthesize(text: string, options: SynthesizeOptions = {}): Promise<Blob> {
    const body: { text: string; language?: string; speed?: number; voice?: string } = { text };
    if (options.language && options.language !== "auto") body.language = options.language;
    if (options.speed !== undefined) body.speed = options.speed;
    if (options.voice) body.voice = options.voice;
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
    form.append("file", file, audioFileName(file));
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

export async function switchModel(category: string, engine: string, model: string): Promise<void> {
    await apiFetch("/api/system/switch-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, engine, model }),
    });
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
    form.append("file", file, audioFileName(file));
    const res = await apiFetch("/api/analyze", { method: "POST", body: form });
    return res.json();
}
