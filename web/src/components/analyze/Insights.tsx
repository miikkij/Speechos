"use client";

import Link from "next/link";
import { Eyebrow, MeterRow, Panel, Skeleton } from "@/components/ui";
import type { AnalysisResult, TranscriptionResult } from "@/lib/api";
import { emotionLabel, formatTime } from "@/lib/format";

interface InsightsProps {
    analysis: AnalysisResult | null;
    transcription: TranscriptionResult | null;
    analyzing: boolean;
    speakers: Map<string, { name: string; color: string }>;
    modelsLine: string | null;
}

/** First line of an API error, trimmed so a stack of Hugging Face details does not fill the card. */
function shortError(message: string): string {
    const first = message.split(/\n|(?<=\.)\s/)[0].trim();
    return first.length > 140 ? `${first.slice(0, 137)}…` : first;
}

/** Words per minute over the transcribed segments — far more reliable than the onset-based estimate. */
function transcriptWpm(transcription: TranscriptionResult | null): number | null {
    if (!transcription?.segments.length) return null;
    let words = 0;
    let seconds = 0;
    for (const seg of transcription.segments) {
        words += seg.text.trim().split(/\s+/).filter(Boolean).length;
        seconds += Math.max(0, seg.end - seg.start);
    }
    return seconds > 1 ? (words / seconds) * 60 : null;
}

function rateNote(wpm: number): string {
    if (wpm < 110) return "slow";
    if (wpm > 170) return "fast";
    return "typical conversation";
}

function summarize(analysis: AnalysisResult, speakers: Map<string, { name: string }>, wpm: number): string {
    const parts: string[] = [];
    const diar = analysis.diarization;
    const n = diar?.summary.num_speakers ?? 0;
    if (n === 1) parts.push("One speaker.");
    if (n > 1) {
        const total = Object.values(diar.summary.speakers).reduce((a, b) => a + b, 0);
        const [topRaw, topSecs] = Object.entries(diar.summary.speakers).sort((a, b) => b[1] - a[1])[0];
        const share = total > 0 ? Math.round((topSecs / total) * 100) : 0;
        parts.push(`${n} speakers. ${speakers.get(topRaw)?.name ?? topRaw} talks ${share}% of the time.`);
    }
    if (analysis.primary_emotion) {
        parts.push(`Delivery is mostly ${emotionLabel(analysis.primary_emotion).toLowerCase()}.`);
    }
    if (wpm > 0) parts.push(`Pace is ${rateNote(wpm)} at about ${Math.round(wpm)} words per minute.`);
    return parts.join(" ");
}

export function Insights({ analysis, transcription, analyzing, speakers, modelsLine }: InsightsProps) {
    if (analyzing && !analysis) {
        return (
            <aside aria-label="Insights" className="flex min-w-0 flex-[1_1_320px] flex-col gap-5">
                <Panel className="gap-4">
                    <Eyebrow>Summary</Eyebrow>
                    <Skeleton lines={3} />
                </Panel>
                <Panel className="gap-4">
                    <Eyebrow>Emotion</Eyebrow>
                    <Skeleton lines={4} />
                </Panel>
            </aside>
        );
    }
    if (!analysis) return null;

    const f = analysis.features;
    const emotions = [...analysis.emotions].sort((a, b) => b.score - a.score).slice(0, 5);
    const diar = analysis.diarization;
    const speakerTotals = diar ? Object.entries(diar.summary.speakers) : [];
    const totalSpeech = speakerTotals.reduce((a, [, s]) => a + s, 0);
    const speechRatio = analysis.speech?.speech_ratio;
    const measuredWpm = transcriptWpm(transcription);
    const wpm = measuredWpm ?? f.speaking_rate.estimated_wpm;
    const summary = summarize(analysis, speakers, wpm);

    const voice = [
        f.pitch.mean_hz != null && {
            label: "Pitch",
            value: `${Math.round(f.pitch.mean_hz)} Hz`,
            note: f.pitch.min_hz != null && f.pitch.max_hz != null ? `range ${Math.round(f.pitch.min_hz)}–${Math.round(f.pitch.max_hz)} Hz` : "",
        },
        {
            label: "Speaking rate",
            value: `${Math.round(wpm)} wpm`,
            note: measuredWpm != null ? `${rateNote(wpm)} · from transcript` : `${rateNote(wpm)} · estimate`,
        },
        { label: "Loudness", value: `${f.energy.mean_db.toFixed(0)} dB`, note: `range ${f.energy.dynamic_range_db.toFixed(0)} dB` },
        {
            label: "Duration",
            value: formatTime(f.duration),
            note: speechRatio != null ? `${Math.round(speechRatio * 100)}% speech` : "",
        },
    ].filter((v): v is { label: string; value: string; note: string } => Boolean(v));

    return (
        <aside aria-label="Insights" className="flex min-w-0 flex-[1_1_320px] flex-col gap-5">
            {summary && (
                <Panel className="gap-3.5">
                    <Eyebrow>Summary</Eyebrow>
                    <p className="m-0 text-[15px] leading-relaxed">
                        {summary}
                        {transcription && transcription.language && (
                            <> Language: {transcription.language.toUpperCase()} ({Math.round(transcription.language_probability * 100)}%).</>
                        )}
                    </p>
                </Panel>
            )}

            <Panel className="gap-3">
                <Eyebrow>Emotion</Eyebrow>
                {analysis.emotion_available && emotions.length > 0 ? (
                    emotions.map((e, i) => (
                        <MeterRow
                            key={e.label}
                            label={emotionLabel(e.label)}
                            value={`${Math.round(e.score * 100)}%`}
                            fraction={e.score}
                            color={i === 0 ? "var(--color-accent)" : "#5b6270"}
                            strong={i === 0}
                        />
                    ))
                ) : (
                    <p className="m-0 text-[13px] text-muted">
                        {analysis.emotion_error ? shortError(analysis.emotion_error) : "Emotion recognition is not enabled for this hardware tier."}{" "}
                        <Link href="/models">Choose a model</Link>
                    </p>
                )}
            </Panel>

            <Panel className="gap-3">
                <Eyebrow>Speakers</Eyebrow>
                {analysis.diarization_available && speakerTotals.length > 0 ? (
                    speakerTotals.map(([raw, secs]) => {
                        const sp = speakers.get(raw);
                        return (
                            <MeterRow
                                key={raw}
                                label={<span className="text-fg">{sp?.name ?? raw}</span>}
                                value={`${formatTime(secs)} · ${totalSpeech > 0 ? Math.round((secs / totalSpeech) * 100) : 0}%`}
                                fraction={totalSpeech > 0 ? secs / totalSpeech : 0}
                                color={sp?.color ?? "var(--color-accent)"}
                                valueWidth="80px"
                            />
                        );
                    })
                ) : (
                    <p className="m-0 text-[13px] text-muted">
                        {diar?.error ? shortError(diar.error) : "Speaker analysis is not enabled for this hardware tier."}{" "}
                        <Link href="/models">Choose a model</Link>
                    </p>
                )}
            </Panel>

            <Panel className="gap-3">
                <Eyebrow>Voice</Eyebrow>
                <div className="grid grid-cols-2 gap-2.5">
                    {voice.map((v) => (
                        <div key={v.label} className="flex flex-col gap-0.5 rounded-[10px] bg-surface-2 p-3">
                            <span className="text-xs text-muted">{v.label}</span>
                            <span className="font-mono text-lg font-semibold">{v.value}</span>
                            <span className="text-xs text-muted">{v.note}</span>
                        </div>
                    ))}
                </div>
                <details className="border-t border-line pt-2.5">
                    <summary className="flex min-h-8 cursor-pointer items-center text-[13px] text-muted">Technical details</summary>
                    <dl className="mt-2 mb-0 grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1.5 text-[13px]">
                        <dt className="text-muted">Tempo</dt>
                        <dd className="m-0 font-mono">{Math.round(f.tempo_bpm)} BPM</dd>
                        <dt className="text-muted">Brightness (spectral centroid)</dt>
                        <dd className="m-0 font-mono">{Math.round(f.spectral.centroid_mean_hz)} Hz</dd>
                        <dt className="text-muted">Syllables per second</dt>
                        <dd className="m-0 font-mono">{f.speaking_rate.syllables_per_sec.toFixed(1)}</dd>
                        <dt className="text-muted">MFCC coefficients</dt>
                        <dd className="m-0 font-mono">{f.mfcc_means.length} values</dd>
                        <dt className="text-muted">Analysis time</dt>
                        <dd className="m-0 font-mono">{analysis.processing_time.toFixed(1)} s</dd>
                    </dl>
                </details>
            </Panel>

            {modelsLine && (
                <p className="m-0 text-xs leading-relaxed text-faint">
                    Models: {modelsLine}. <Link href="/models">Change models</Link>
                </p>
            )}
        </aside>
    );
}
