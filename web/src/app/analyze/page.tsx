"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Insights } from "@/components/analyze/Insights";
import { RecordingsSidebar } from "@/components/analyze/RecordingsSidebar";
import { Transcript } from "@/components/analyze/Transcript";
import { DownloadIcon, PauseIcon, PlayIcon, RefreshIcon, TrashIcon } from "@/components/Icons";
import { Panel, buttonSecondary, fieldClass } from "@/components/ui";
import { Waveform } from "@/components/Waveform";
import { useAudioElement } from "@/hooks/useAudioElement";
import { usePeaks } from "@/hooks/usePeaks";
import { deleteRecording, getModelOptions, getRecordings, type Recording } from "@/lib/api";
import { assignSpeakers, baseName, downloadText, formatTime, speakerPalette, toPlainText, toSrt } from "@/lib/format";
import { useApp } from "@/lib/store";

const BUCKETS = 140;
const RATES = [0.75, 1, 1.25, 1.5];

interface Lane {
    label: string;
    items: { start: number; end: number; text: string; color: string; textColor: string }[];
}

function LaneRow({ lane, duration }: { lane: Lane; duration: number }) {
    return (
        <>
            <span className="text-xs text-muted">{lane.label}</span>
            <div className="relative h-[22px] overflow-hidden rounded-md bg-surface-2">
                {lane.items.map((it, i) => (
                    <span
                        key={i}
                        title={`${it.text} · ${formatTime(it.start)}–${formatTime(it.end)}`}
                        className="absolute top-0 bottom-0 flex items-center overflow-hidden border-r border-surface pl-1.5 text-[11px] font-semibold whitespace-nowrap"
                        style={{
                            left: `${(it.start / duration) * 100}%`,
                            width: `${((it.end - it.start) / duration) * 100}%`,
                            background: it.color,
                            color: it.textColor,
                        }}
                    >
                        {it.text}
                    </span>
                ))}
            </div>
        </>
    );
}

function AnalyzeWorkspace() {
    const params = useSearchParams();
    const router = useRouter();
    const recordingId = params.get("r");
    const { results, runAnalysis, bumpRecordings } = useApp();
    const [loaded, setLoaded] = useState<{ id: string; recording: Recording | null } | null>(null);
    const recording = loaded && loaded.id === recordingId ? loaded.recording : null;
    const [modelsLine, setModelsLine] = useState<string | null>(null);

    const result = recordingId ? results[recordingId] : undefined;
    // Changes whenever a new analysis result lands; used to refresh the "Models:" line
    const analysedAt = result?.analysis?.processing_time ?? null;
    const blob = result?.blob ?? null;
    const peaks = usePeaks(blob, BUCKETS);
    const { attach: attachAudio, ...audio } = useAudioElement(blob);

    useEffect(() => {
        if (recordingId) runAnalysis(recordingId);
    }, [recordingId, runAnalysis]);

    useEffect(() => {
        if (!recordingId) return;
        getRecordings()
            .then((recs) => setLoaded({ id: recordingId, recording: recs.find((r) => r.id === recordingId) ?? null }))
            .catch(() => setLoaded({ id: recordingId, recording: null }));
    }, [recordingId]);

    useEffect(() => {
        getModelOptions()
            .then((data) => {
                const label = (cat: string) => {
                    const cur = data.current[cat];
                    if (!cur) return null;
                    const opt = data.options[cat]?.find((o) => o.engine === cur.engine && o.model === cur.model);
                    return opt ? opt.label.split(" (")[0] : `${cur.engine} ${cur.model}`;
                };
                setModelsLine(["stt", "emotion", "diarization"].map(label).filter(Boolean).join(" · ") || null);
            })
            .catch(() => setModelsLine(null));
        // Re-read after each analysis: the API may switch a model (e.g. gated pyannote → Resemblyzer)
    }, [analysedAt]);

    const transcription = result?.transcription ?? null;
    const analysis = result?.analysis ?? null;
    const duration = audio.duration || peaks?.duration || transcription?.duration || analysis?.features.duration || 0;

    const speakers = useMemo(
        () => speakerPalette(analysis?.diarization?.segments.map((s) => s.speaker) ?? []),
        [analysis],
    );
    const lines = useMemo(
        () => (transcription ? assignSpeakers(transcription.segments, analysis?.diarization?.segments) : null),
        [transcription, analysis],
    );

    const lanes: Lane[] = [];
    if (analysis?.diarization?.segments.length) {
        lanes.push({
            label: "Speakers",
            items: analysis.diarization.segments.map((s) => {
                const sp = speakers.get(s.speaker);
                return {
                    start: s.start,
                    end: s.end,
                    text: sp ? sp.name.replace("Speaker ", "S") : s.speaker,
                    color: sp?.color ?? "#5b6270",
                    textColor: "var(--color-bg)",
                };
            }),
        });
    }
    if (analysis?.speech?.segments.length) {
        lanes.push({
            label: "Speech",
            items: analysis.speech.segments.map((s) => ({
                start: s.start,
                end: s.end,
                text: "",
                color: "#3a4150",
                textColor: "var(--color-fg)",
            })),
        });
    }

    const speechSegments = analysis?.speech?.segments;
    const speechMask =
        speechSegments?.length && duration
            ? Array.from({ length: BUCKETS }, (_, i) => {
                  const t = ((i + 0.5) / BUCKETS) * duration;
                  return speechSegments.some((s) => t >= s.start - 0.15 && t <= s.end + 0.15);
              })
            : undefined;

    const name = recording?.filename || recordingId || "";
    const meta = [
        duration ? formatTime(duration) : null,
        "16 kHz mono",
        transcription ? `${transcription.language.toUpperCase()} (${Math.round(transcription.language_probability * 100)}%)` : null,
        analysis ? `analyzed in ${analysis.processing_time.toFixed(1)} s` : null,
    ]
        .filter(Boolean)
        .join(" · ");

    const exportAs = (kind: "txt" | "srt" | "json") => {
        const stem = baseName(name) || "recording";
        if (kind === "json") {
            downloadText(`${stem}.json`, JSON.stringify({ recording, transcription, analysis }, null, 2), "application/json");
        } else if (lines) {
            const content = kind === "txt" ? toPlainText(lines, speakers) : toSrt(lines, speakers);
            downloadText(`${stem}.${kind}`, content, "text/plain");
        }
    };

    const remove = async () => {
        if (!recordingId || !window.confirm(`Delete ${name}? This cannot be undone.`)) return;
        await deleteRecording(recordingId).catch(() => undefined);
        bumpRecordings();
        router.push("/analyze");
    };

    return (
        <div className="mx-auto flex max-w-[1560px] flex-wrap items-start gap-5 p-6">
            <RecordingsSidebar selectedId={recordingId} />

            <main className="flex min-w-0 flex-[999_1_560px] flex-col gap-5">
                {!recordingId && (
                    <Panel className="items-start gap-3 py-10">
                        <h1 className="m-0 text-xl font-semibold">Pick a recording</h1>
                        <p className="m-0 text-muted">
                            Choose one from the list, or <Link href="/capture">record or upload a new one</Link>.
                        </p>
                    </Panel>
                )}

                {recordingId && (
                    <>
                        <Panel label="Recording" className="gap-[18px]">
                            <div className="flex flex-wrap items-start justify-between gap-3">
                                <div className="min-w-0">
                                    <h1 className="m-0 truncate text-xl font-semibold tracking-tight">{name}</h1>
                                    <p className="mt-1 mb-0 text-[13px] text-muted">{meta}</p>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    <button
                                        type="button"
                                        onClick={() => runAnalysis(recordingId, true)}
                                        disabled={result?.transcribing || result?.analyzing}
                                        className={buttonSecondary}
                                    >
                                        <RefreshIcon />
                                        {result?.transcribing || result?.analyzing ? "Analyzing…" : "Re-analyze"}
                                    </button>
                                    <details className="relative">
                                        <summary className={`${buttonSecondary} cursor-pointer list-none`}>
                                            <DownloadIcon />
                                            Export
                                        </summary>
                                        <div className="absolute right-0 z-10 mt-1 flex w-48 flex-col rounded-lg border border-line-2 bg-surface-2 p-1 shadow-lg">
                                            {(
                                                [
                                                    ["txt", "Transcript (.txt)"],
                                                    ["srt", "Subtitles (.srt)"],
                                                    ["json", "All results (.json)"],
                                                ] as const
                                            ).map(([kind, label]) => (
                                                <button
                                                    key={kind}
                                                    type="button"
                                                    onClick={() => exportAs(kind)}
                                                    disabled={kind !== "json" && !lines}
                                                    className="rounded-md px-3 py-2 text-left hover:bg-nav disabled:opacity-50"
                                                >
                                                    {label}
                                                </button>
                                            ))}
                                        </div>
                                    </details>
                                    <button type="button" onClick={remove} aria-label="Delete recording" className={buttonSecondary}>
                                        <TrashIcon />
                                    </button>
                                </div>
                            </div>

                            {result?.error && (
                                <p role="alert" className="m-0 text-sm text-bad">
                                    {result.error}
                                </p>
                            )}

                            <div className="flex items-center gap-3.5">
                                <button
                                    type="button"
                                    onClick={audio.toggle}
                                    disabled={!audio.src}
                                    aria-label={audio.playing ? "Pause" : "Play"}
                                    className="flex h-11 w-11 flex-none items-center justify-center rounded-full bg-accent text-bg hover:bg-accent-hover disabled:opacity-50"
                                >
                                    {audio.playing ? <PauseIcon /> : <PlayIcon />}
                                </button>
                                <span className="font-mono text-[13px] text-muted">
                                    <span className="text-fg">{formatTime(audio.currentTime)}</span> / {formatTime(duration)}
                                </span>
                                <label className="ml-auto flex items-center gap-1.5 text-xs text-muted">
                                    Speed
                                    <select
                                        value={audio.rate}
                                        onChange={(e) => audio.setRate(Number(e.target.value))}
                                        className={`${fieldClass} px-2 py-1.5 text-xs`}
                                    >
                                        {RATES.map((r) => (
                                            <option key={r} value={r}>
                                                {r}×
                                            </option>
                                        ))}
                                    </select>
                                </label>
                                <audio ref={attachAudio} src={audio.src} preload="metadata" {...audio.bind} />
                            </div>

                            <div className="flex flex-col gap-2">
                                <Waveform
                                    values={peaks?.values ?? null}
                                    duration={duration}
                                    currentTime={audio.currentTime}
                                    onSeek={(t) => audio.seek(t)}
                                    speech={speechMask}
                                />
                                {duration > 0 && (
                                    <div className="grid grid-cols-[72px_minmax(0,1fr)] items-center gap-2.5">
                                        {lanes.map((lane) => (
                                            <LaneRow key={lane.label} lane={lane} duration={duration} />
                                        ))}
                                        <span />
                                        <div className="relative h-4 font-mono text-[11px] text-faint" aria-hidden="true">
                                            {[0, 0.25, 0.5, 0.75].map((f) => (
                                                <span key={f} className="absolute top-0" style={{ left: `${f * 100}%` }}>
                                                    {formatTime(f * duration)}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </Panel>

                        <Transcript
                            lines={lines}
                            loading={Boolean(result?.transcribing)}
                            currentTime={audio.currentTime}
                            speakers={speakers}
                            onSeek={(t) => audio.seek(t, true)}
                        />
                    </>
                )}
            </main>

            {recordingId && (
                <Insights
                    analysis={analysis}
                    transcription={transcription}
                    analyzing={Boolean(result?.analyzing)}
                    speakers={speakers}
                    modelsLine={modelsLine}
                />
            )}
        </div>
    );
}

export default function AnalyzePage() {
    return (
        <Suspense fallback={null}>
            <AnalyzeWorkspace />
        </Suspense>
    );
}
