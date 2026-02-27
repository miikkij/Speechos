"use client";

import type { TranscriptionResult } from "@/lib/api";

interface TranscriptionViewProps {
    result: TranscriptionResult | null;
    loading?: boolean;
}

export function TranscriptionView({ result, loading }: TranscriptionViewProps) {
    if (loading) {
        return (
            <div className="rounded-lg p-6 animate-pulse" style={{ background: "var(--surface)" }}>
                <div className="h-4 rounded w-3/4 mb-3" style={{ background: "var(--border)" }} />
                <div className="h-4 rounded w-1/2" style={{ background: "var(--border)" }} />
            </div>
        );
    }

    if (!result) return null;

    return (
        <div className="rounded-lg p-6" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-3 mb-4">
                <h3 className="text-sm font-medium" style={{ color: "var(--muted)" }}>
                    Transcription
                </h3>
                <span
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: "var(--surface-2)", color: "var(--muted)" }}
                >
                    {result.language} ({Math.round(result.language_probability * 100)}%)
                </span>
                <span
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: "var(--surface-2)", color: "var(--muted)" }}
                >
                    {result.processing_time}s
                </span>
            </div>

            <p className="text-base leading-relaxed mb-4">{result.text}</p>

            {result.segments.length > 0 && (
                <details>
                    <summary className="text-xs cursor-pointer mb-2" style={{ color: "var(--muted)" }}>
                        {result.segments.length} segments
                    </summary>
                    <div className="space-y-1">
                        {result.segments.map((seg, i) => (
                            <div key={i} className="flex gap-2 text-xs">
                                <span className="font-mono shrink-0" style={{ color: "var(--accent)" }}>
                                    {seg.start.toFixed(1)}–{seg.end.toFixed(1)}s
                                </span>
                                <span>{seg.text}</span>
                            </div>
                        ))}
                    </div>
                </details>
            )}
        </div>
    );
}
