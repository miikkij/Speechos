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

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
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

export async function synthesize(text: string): Promise<Blob> {
  const res = await apiFetch("/api/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
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

export async function getSystemInfo(): Promise<SystemInfo> {
  const res = await apiFetch("/api/system/info");
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
