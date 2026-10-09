"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { MicIcon, StopIcon, UploadIcon } from "@/components/Icons";
import { LanguageSelect } from "@/components/LanguageSelect";
import { Eyebrow, fieldClass } from "@/components/ui";
import { useAudioDevices } from "@/hooks/useAudioDevices";
import { useRecorder } from "@/hooks/useRecorder";
import { getRecordings, uploadRecording, type Recording } from "@/lib/api";
import { formatDate, formatTime } from "@/lib/format";
import { useApp } from "@/lib/store";

export default function CapturePage() {
    const router = useRouter();
    const { language, setLanguage, options, setOptions, bumpRecordings, recordingsVersion, results } = useApp();
    const { devices, selectedId, setSelectedId } = useAudioDevices();
    const { isRecording, duration, start, stop, levels } = useRecorder(selectedId || undefined);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dragging, setDragging] = useState(false);
    const [recent, setRecent] = useState<Recording[]>([]);
    const fileInput = useRef<HTMLInputElement>(null);

    useEffect(() => {
        getRecordings()
            .then((recs) => setRecent([...recs].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 3)))
            .catch(() => setRecent([]));
    }, [recordingsVersion]);

    const save = useCallback(
        async (blob: Blob) => {
            setSaving(true);
            setError(null);
            try {
                const rec = await uploadRecording(blob);
                bumpRecordings();
                router.push(`/analyze?r=${encodeURIComponent(rec.id)}`);
            } catch (err) {
                setError(err instanceof Error ? err.message : "Saving the recording failed");
                setSaving(false);
            }
        },
        [bumpRecordings, router],
    );

    const toggle = async () => {
        setError(null);
        if (isRecording) {
            const blob = await stop();
            await save(blob);
            return;
        }
        try {
            await start();
        } catch {
            setError("Microphone access was blocked. Allow it in the browser and try again.");
        }
    };

    const onFiles = (files: FileList | null) => {
        const file = files?.[0];
        if (file) save(file);
    };

    let hint = "Press to start recording";
    if (isRecording) hint = "Recording — press to stop and analyze";
    if (saving) hint = "Saving…";

    return (
        <div className="mx-auto flex max-w-[980px] flex-col gap-6 px-6 py-10">
            <div>
                <h1 className="m-0 text-[26px] font-semibold tracking-tight">Capture speech</h1>
                <p className="mt-1.5 mb-0 text-[15px] text-muted">
                    Record from a microphone or add a file. Everything is processed on this machine.
                </p>
            </div>

            <section
                aria-label="Record"
                className="flex flex-col items-center gap-5 rounded-2xl border border-line bg-surface px-6 py-8"
            >
                <button
                    type="button"
                    onClick={toggle}
                    disabled={saving}
                    aria-label={isRecording ? "Stop recording" : "Start recording"}
                    className={`flex h-24 w-24 items-center justify-center rounded-full border-[6px] text-bg disabled:opacity-60 ${
                        isRecording ? "border-[#3a1e1a] bg-rec" : "border-nav bg-fg hover:bg-white"
                    }`}
                >
                    {isRecording ? <StopIcon /> : <MicIcon />}
                </button>
                <div className="flex flex-col items-center gap-1">
                    <span className="font-mono text-[28px] font-medium tracking-wide" aria-live="off">
                        {formatTime(duration).padStart(5, "0")}
                    </span>
                    <span className="text-[13px] text-muted" aria-live="polite">
                        {hint}
                    </span>
                </div>

                <div aria-hidden="true" className="flex h-10 w-full max-w-[520px] items-center gap-[3px]">
                    {levels.map((lv, i) => (
                        <span
                            key={i}
                            className={`flex-1 rounded-sm ${isRecording ? "bg-accent" : "bg-line-2"}`}
                            style={{ height: `${Math.round(4 + Math.min(1, lv * 2.2) * 32)}px` }}
                        />
                    ))}
                </div>

                <div className="grid w-full max-w-[640px] gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
                    <label className="flex flex-col gap-1.5 text-xs text-muted">
                        Microphone
                        <select
                            value={selectedId}
                            onChange={(e) => setSelectedId(e.target.value)}
                            disabled={isRecording}
                            className={`${fieldClass} p-2.5 text-sm`}
                        >
                            {devices.length === 0 && <option value="">Default input</option>}
                            {devices.map((d) => (
                                <option key={d.deviceId} value={d.deviceId}>
                                    {d.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <LanguageSelect value={language} onChange={setLanguage} label="Spoken language" layout="stacked" />
                </div>

                <fieldset className="m-0 flex w-full max-w-[640px] flex-wrap gap-x-6 gap-y-2.5 rounded-xl border border-line px-4 py-3.5">
                    <legend className="px-1.5 text-xs text-muted">After recording, run</legend>
                    <label className="flex min-h-8 items-center gap-2">
                        <input
                            type="checkbox"
                            checked={options.transcript}
                            onChange={(e) => setOptions({ ...options, transcript: e.target.checked })}
                            className="h-[18px] w-[18px] accent-accent"
                        />
                        Transcript
                    </label>
                    <label className="flex min-h-8 items-center gap-2">
                        <input
                            type="checkbox"
                            checked={options.voice}
                            onChange={(e) => setOptions({ ...options, voice: e.target.checked })}
                            className="h-[18px] w-[18px] accent-accent"
                        />
                        Voice analysis (emotion, speakers, features)
                    </label>
                </fieldset>

                {error && (
                    <p role="alert" className="m-0 text-sm text-bad">
                        {error}
                    </p>
                )}
            </section>

            <div className="flex items-center gap-3 text-xs text-faint">
                <span className="h-px flex-1 bg-line" />
                or
                <span className="h-px flex-1 bg-line" />
            </div>

            <section
                aria-label="Upload"
                onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    onFiles(e.dataTransfer.files);
                }}
                className={`flex flex-wrap items-center justify-center gap-4 rounded-2xl border-[1.5px] border-dashed px-6 py-7 text-center ${
                    dragging ? "border-accent bg-selected" : "border-line-3"
                }`}
            >
                <UploadIcon className="text-muted" />
                <div className="flex flex-col gap-0.5">
                    <span className="text-[15px] font-medium">Drop an audio file here</span>
                    <span className="text-[13px] text-muted">WAV, MP3, FLAC, OGG or WebM · converted to 16 kHz mono WAV</span>
                </div>
                <button
                    type="button"
                    onClick={() => fileInput.current?.click()}
                    disabled={saving}
                    className="inline-flex min-h-11 items-center rounded-lg border border-line-2 bg-surface-2 px-4 font-medium hover:border-line-3"
                >
                    Choose file
                </button>
                <input
                    ref={fileInput}
                    type="file"
                    accept="audio/*"
                    className="sr-only"
                    tabIndex={-1}
                    onChange={(e) => onFiles(e.target.files)}
                />
            </section>

            {recent.length > 0 && (
                <section aria-label="Recent recordings" className="flex flex-col gap-2.5">
                    <div className="flex items-baseline justify-between">
                        <Eyebrow>Recent</Eyebrow>
                        <Link href="/analyze" className="text-[13px]">
                            All recordings
                        </Link>
                    </div>
                    <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
                        {recent.map((r) => {
                            const snippet = results[r.id]?.transcription?.text;
                            return (
                                <Link
                                    key={r.id}
                                    href={`/analyze?r=${encodeURIComponent(r.id)}`}
                                    className="flex min-w-0 flex-col gap-1 rounded-xl border border-line bg-surface p-3.5 text-fg no-underline hover:border-line-3 hover:text-fg"
                                >
                                    <span className="truncate font-medium">{r.filename || r.id}</span>
                                    <span className="font-mono text-xs text-muted">
                                        {formatDate(r.created_at)} · {formatTime(r.duration)}
                                    </span>
                                    <span className="truncate text-[13px] text-muted">{snippet ?? "Open to analyze"}</span>
                                </Link>
                            );
                        })}
                    </div>
                </section>
            )}
        </div>
    );
}
