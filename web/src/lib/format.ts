import type { DiarizationSegment, TranscriptionResult } from "@/lib/api";

/** 83.4 → "1:23" */
export function formatTime(seconds: number): string {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** 83.456 → "00:01:23,456" (SRT) */
function srtTime(seconds: number): string {
    const ms = Math.max(0, Math.round(seconds * 1000));
    const h = Math.floor(ms / 3_600_000);
    const m = Math.floor((ms % 3_600_000) / 60_000);
    const s = Math.floor((ms % 60_000) / 1000);
    const rest = ms % 1000;
    const p = (n: number, w = 2) => String(n).padStart(w, "0");
    return `${p(h)}:${p(m)}:${p(s)},${p(rest, 3)}`;
}

export function formatDate(iso: string): string {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return `${d.getDate()}.${d.getMonth() + 1}.`;
}

/** "生气/angry" or "ANGRY" → "Angry" */
export function emotionLabel(raw: string): string {
    const part = raw.includes("/") ? raw.split("/").pop() ?? raw : raw;
    const clean = part.replace(/[^a-zA-Z ]/g, "").trim();
    if (!clean) return raw;
    if (/^(unk|unknown|other)$/i.test(clean)) return "Other";
    return clean.charAt(0).toUpperCase() + clean.slice(1).toLowerCase();
}

export const SPEAKER_COLORS = ["#8ab4ff", "#e8954a", "#b9e07a", "#c9a7ff", "#7fd1c7", "#f2b84b"];

/** Stable "Speaker N" names and colors for raw diarization labels (SPEAKER_00 …). */
export function speakerPalette(rawLabels: string[]): Map<string, { name: string; color: string }> {
    const sorted = [...new Set(rawLabels)].sort();
    return new Map(
        sorted.map((raw, i) => [raw, { name: `Speaker ${i + 1}`, color: SPEAKER_COLORS[i % SPEAKER_COLORS.length] }]),
    );
}

export interface TranscriptLine {
    start: number;
    end: number;
    text: string;
    /** Raw diarization label with the largest overlap, if any. */
    speaker: string | null;
}

export function assignSpeakers(
    segments: TranscriptionResult["segments"],
    diarization: DiarizationSegment[] | undefined,
): TranscriptLine[] {
    return segments.map((seg) => {
        let best: string | null = null;
        let bestOverlap = 0;
        for (const d of diarization ?? []) {
            const overlap = Math.min(seg.end, d.end) - Math.max(seg.start, d.start);
            if (overlap > bestOverlap) {
                bestOverlap = overlap;
                best = d.speaker;
            }
        }
        return { start: seg.start, end: seg.end, text: seg.text.trim(), speaker: best };
    });
}

export function toPlainText(lines: TranscriptLine[], names: Map<string, { name: string }>): string {
    return lines
        .map((l) => {
            const who = l.speaker ? names.get(l.speaker)?.name : null;
            return `[${formatTime(l.start)}]${who ? ` ${who}:` : ""} ${l.text}`;
        })
        .join("\n");
}

export function toSrt(lines: TranscriptLine[], names: Map<string, { name: string }>): string {
    return lines
        .map((l, i) => {
            const who = l.speaker ? names.get(l.speaker)?.name : null;
            return `${i + 1}\n${srtTime(l.start)} --> ${srtTime(l.end)}\n${who ? `${who}: ` : ""}${l.text}\n`;
        })
        .join("\n");
}

export function downloadText(filename: string, content: string, type: string): void {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/** Sample rate from a WAV (RIFF) header, or null for other formats. */
export async function wavSampleRate(blob: Blob): Promise<number | null> {
    const view = new DataView(await blob.slice(0, 28).arrayBuffer());
    if (view.byteLength < 28 || view.getUint32(0, false) !== 0x52494646) return null; // "RIFF"
    return view.getUint32(24, true);
}

/** A local model path → its readable name: ".../whisper-large-finnish-v3-ct2-f16" → "whisper-large-finnish-v3" */
export function modelDisplayName(model: string): string {
    const last = model.split(/[\\/]/).filter(Boolean).pop() ?? model;
    return last.replace(/-ct2(-[a-z0-9]+)?$/i, "");
}

/** "recording_2026-10-08.wav" → "recording_2026-10-08" */
export function baseName(name: string): string {
    return name.replace(/\.[a-z0-9]+$/i, "");
}
