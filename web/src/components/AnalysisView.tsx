"use client";

import type { AnalysisResult } from "@/lib/api";

interface AnalysisViewProps {
    result: AnalysisResult | null;
    loading?: boolean;
}

const SPEAKER_COLORS = ["#f59e0b", "#3b82f6", "#22c55e", "#ef4444", "#a855f7", "#ec4899", "#14b8a6", "#f97316"];

const EMOTION_COLORS: Record<string, string> = {
    happy: "#22c55e",
    angry: "#ef4444",
    sad: "#3b82f6",
    neutral: "#9ca3af",
    surprised: "#f59e0b",
    fearful: "#a855f7",
    disgusted: "#84cc16",
    unknown: "#6b7280",
};

function getEmotionColor(label: string): string {
    const key = label.toLowerCase().replace(/[^a-z]/g, "");
    for (const [k, v] of Object.entries(EMOTION_COLORS)) {
        if (key.includes(k)) return v;
    }
    return EMOTION_COLORS.unknown;
}

export function AnalysisView({ result, loading }: AnalysisViewProps) {
    if (loading) {
        return (
            <div className="rounded-lg p-6 animate-pulse" style={{ background: "var(--surface)" }}>
                <div className="h-4 rounded w-1/2 mb-3" style={{ background: "var(--border)" }} />
                <div className="h-4 rounded w-2/3 mb-3" style={{ background: "var(--border)" }} />
                <div className="h-4 rounded w-1/3" style={{ background: "var(--border)" }} />
            </div>
        );
    }

    if (!result) return null;

    const { features, emotions, primary_emotion, emotion_available, diarization, diarization_available } = result;

    return (
        <div className="rounded-lg p-6 space-y-5" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            {/* Header */}
            <div className="flex items-center gap-3">
                <h3 className="text-sm font-medium" style={{ color: "var(--muted)" }}>
                    Audio Analysis
                </h3>
                <span
                    className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: "var(--surface-2)", color: "var(--muted)" }}
                >
                    {result.processing_time}s
                </span>
            </div>

            {/* Primary emotion badge */}
            {primary_emotion && (
                <div className="flex items-center gap-2">
                    <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                        Detected Emotion
                    </span>
                    <span
                        className="text-sm font-semibold px-3 py-1 rounded-full"
                        style={{
                            background: getEmotionColor(primary_emotion) + "22",
                            color: getEmotionColor(primary_emotion),
                            border: `1px solid ${getEmotionColor(primary_emotion)}44`,
                        }}
                    >
                        {primary_emotion}
                    </span>
                </div>
            )}

            {/* Emotion scores bar chart */}
            {emotions.length > 0 && (
                <div className="space-y-1.5">
                    {emotions.slice(0, 5).map((e) => (
                        <div key={e.label} className="flex items-center gap-2 text-xs">
                            <span className="w-20 shrink-0 text-right" style={{ color: "var(--muted)" }}>
                                {e.label}
                            </span>
                            <div className="flex-1 h-3 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
                                <div
                                    className="h-full rounded-full transition-all"
                                    style={{
                                        width: `${Math.max(e.score * 100, 1)}%`,
                                        background: getEmotionColor(e.label),
                                        opacity: 0.8,
                                    }}
                                />
                            </div>
                            <span className="w-12 shrink-0 font-mono" style={{ color: "var(--muted)" }}>
                                {(e.score * 100).toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* Audio features grid */}
            <div className="grid grid-cols-2 gap-3">
                {/* Pitch */}
                {features.pitch.mean_hz != null && (
                    <FeatureCard
                        label="Pitch"
                        value={`${features.pitch.mean_hz} Hz`}
                        detail={`${features.pitch.min_hz}–${features.pitch.max_hz} Hz`}
                    />
                )}

                {/* Energy */}
                <FeatureCard
                    label="Loudness"
                    value={`${features.energy.mean_db} dB`}
                    detail={`range: ${features.energy.dynamic_range_db} dB`}
                />

                {/* Speaking Rate */}
                <FeatureCard
                    label="Speaking Rate"
                    value={`~${features.speaking_rate.estimated_wpm} wpm`}
                    detail={`${features.speaking_rate.syllables_per_sec} syl/s`}
                />

                {/* Tempo */}
                <FeatureCard
                    label="Tempo"
                    value={`${features.tempo_bpm} BPM`}
                />

                {/* Spectral */}
                <FeatureCard
                    label="Brightness"
                    value={`${features.spectral.centroid_mean_hz} Hz`}
                    detail="spectral centroid"
                />

                {/* Duration */}
                <FeatureCard
                    label="Duration"
                    value={`${features.duration}s`}
                />
            </div>

            {/* Speaker diarization */}
            {diarization && diarization.summary.num_speakers > 0 && (
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                            Speakers Detected
                        </span>
                        <span
                            className="text-xs px-2 py-0.5 rounded-full font-semibold"
                            style={{ background: "var(--surface-2)", color: "var(--accent)" }}
                        >
                            {diarization.summary.num_speakers}
                        </span>
                    </div>
                    <div className="space-y-1">
                        {Object.entries(diarization.summary.speakers).map(([speaker, duration]) => (
                            <div key={speaker} className="flex items-center gap-2 text-xs">
                                <span
                                    className="w-20 shrink-0 text-right font-mono"
                                    style={{ color: SPEAKER_COLORS[Object.keys(diarization.summary.speakers).indexOf(speaker) % SPEAKER_COLORS.length] }}
                                >
                                    {speaker}
                                </span>
                                <div className="flex-1 h-3 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
                                    <div
                                        className="h-full rounded-full"
                                        style={{
                                            width: `${Math.max((duration / features.duration) * 100, 2)}%`,
                                            background: SPEAKER_COLORS[Object.keys(diarization.summary.speakers).indexOf(speaker) % SPEAKER_COLORS.length],
                                            opacity: 0.7,
                                        }}
                                    />
                                </div>
                                <span className="w-12 shrink-0 font-mono" style={{ color: "var(--muted)" }}>
                                    {duration.toFixed(1)}s
                                </span>
                            </div>
                        ))}
                    </div>
                    {diarization.segments.length > 0 && (
                        <details>
                            <summary className="text-xs cursor-pointer mb-2" style={{ color: "var(--muted)" }}>
                                {diarization.segments.length} speaker turns
                            </summary>
                            <div className="space-y-1">
                                {diarization.segments.map((seg, i) => (
                                    <div key={i} className="flex gap-2 text-xs">
                                        <span
                                            className="font-mono shrink-0"
                                            style={{ color: SPEAKER_COLORS[Object.keys(diarization.summary.speakers).indexOf(seg.speaker) % SPEAKER_COLORS.length] }}
                                        >
                                            {seg.speaker}
                                        </span>
                                        <span className="font-mono shrink-0" style={{ color: "var(--accent)" }}>
                                            {seg.start.toFixed(1)}–{seg.end.toFixed(1)}s
                                        </span>
                                        <span style={{ color: "var(--muted)" }}>({seg.duration.toFixed(1)}s)</span>
                                    </div>
                                ))}
                            </div>
                        </details>
                    )}
                </div>
            )}

            {/* MFCC details (collapsible) */}
            <details>
                <summary className="text-xs cursor-pointer mb-2" style={{ color: "var(--muted)" }}>
                    MFCC coefficients ({features.mfcc_means.length})
                </summary>
                <p className="text-xs mb-2" style={{ color: "var(--muted)" }}>
                    Mel-Frequency Cepstral Coefficients: a compact representation of the audio spectrum
                    that captures vocal characteristics like timbre, pitch shape, and speaker identity.
                    C0 = overall energy, C1-C4 = broad spectral shape, C5-C12 = finer vocal tract detail.
                </p>
                <div className="flex gap-1 flex-wrap">
                    {features.mfcc_means.map((v, i) => (
                        <span
                            key={i}
                            className="text-xs font-mono px-1.5 py-0.5 rounded"
                            style={{ background: "var(--surface-2)", color: "var(--muted)" }}
                            title={`C${i}: ${i === 0 ? "overall energy" : i <= 4 ? "broad spectral shape" : "vocal tract detail"}`}
                        >
                            C{i}: {v.toFixed(1)}
                        </span>
                    ))}
                </div>
            </details>

            {/* Status notes */}
            {(!emotion_available || !diarization_available) && (
                <div className="text-xs space-y-1" style={{ color: "var(--muted)" }}>
                    {!emotion_available && (
                        <p>Emotion detection unavailable (requires GPU + funasr)</p>
                    )}
                    {!diarization_available && (
                        <p>Speaker diarization unavailable (requires GPU + pyannote.audio)</p>
                    )}
                </div>
            )}
        </div>
    );
}

function FeatureCard({ label, value, detail }: { label: string; value: string; detail?: string }) {
    return (
        <div className="rounded-md p-3" style={{ background: "var(--surface-2)" }}>
            <div className="text-xs mb-1" style={{ color: "var(--muted)" }}>{label}</div>
            <div className="text-sm font-semibold">{value}</div>
            {detail && (
                <div className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>{detail}</div>
            )}
        </div>
    );
}
