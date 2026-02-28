"use client";

import { useCallback, useState } from "react";
import { Recorder } from "@/components/Recorder";
import { AudioPlayer } from "@/components/AudioPlayer";
import { TranscriptionView } from "@/components/TranscriptionView";
import { AnalysisView } from "@/components/AnalysisView";
import { TtsPlayground } from "@/components/TtsPlayground";
import { StatusBar } from "@/components/StatusBar";
import { RecordingsLibrary } from "@/components/RecordingsLibrary";
import { ModelSelectors } from "@/components/ModelSelectors";
import { useAudioDevices } from "@/hooks/useAudioDevices";
import { transcribe, analyze, uploadRecording, type TranscriptionResult, type AnalysisResult, type Recording } from "@/lib/api";

type Tab = "record" | "tts";

export default function Home() {
    const [tab, setTab] = useState<Tab>("record");
    const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
    const [recordingLabel, setRecordingLabel] = useState<string>("Recording");
    const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
    const [transcribing, setTranscribing] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [refreshRecordings, setRefreshRecordings] = useState(0);
    const { devices, selectedId, setSelectedId } = useAudioDevices();

    const processAudio = useCallback(async (blob: Blob, label: string, shouldUpload: boolean) => {
        setRecordingBlob(blob);
        setRecordingLabel(label);
        setTranscription(null);
        setAnalysis(null);
        setError(null);
        setTranscribing(true);
        setAnalyzing(true);
        try {
            const promises: Promise<unknown>[] = [transcribe(blob)];
            if (shouldUpload) {
                promises.push(
                    uploadRecording(blob)
                        .then(() => setRefreshRecordings((n) => n + 1))
                        .catch(() => null)
                );
            }
            const [transcribeResult] = await Promise.all(promises);
            setTranscription(transcribeResult as TranscriptionResult);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Transcription failed");
        } finally {
            setTranscribing(false);
        }
        try {
            const analysisResult = await analyze(blob);
            setAnalysis(analysisResult);
        } catch {
            // Analysis failure is non-critical
        } finally {
            setAnalyzing(false);
        }
    }, []);

    const handleRecorded = useCallback((blob: Blob) => {
        processAudio(blob, "Recording", true);
    }, [processAudio]);

    const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        processAudio(file, file.name, true);
    }, [processAudio]);

    const handleSelectRecording = useCallback((blob: Blob, recording: Recording) => {
        processAudio(blob, recording.id, false);
    }, [processAudio]);

    return (
        <div className="flex flex-col min-h-screen relative">
            {/* Ambient stage glow */}
            <div className="ambient-glow" />

            <StatusBar />

            {/* Header */}
            <header className="px-6 pt-8 pb-6 text-center relative z-10">
                <h1
                    className="text-3xl font-bold tracking-widest uppercase"
                    style={{
                        fontFamily: "var(--font-display), serif",
                        color: "var(--accent)",
                        textShadow: "0 0 40px rgba(232, 145, 58, 0.3), 0 2px 4px rgba(0,0,0,0.3)",
                    }}
                >
                    Speechos
                </h1>
                <p
                    className="text-xs mt-2 uppercase tracking-[0.25em]"
                    style={{ color: "var(--muted)" }}
                >
                    Local-first speech analysis & synthesis
                </p>
            </header>

            {/* Tabs */}
            <nav className="px-6 flex justify-center gap-8 relative z-10">
                {(["record", "tts"] as const).map((t) => (
                    <button
                        key={t}
                        onClick={() => setTab(t)}
                        className="relative px-1 py-3 text-sm tracking-wide uppercase transition-colors"
                        style={{
                            color: tab === t ? "var(--accent)" : "var(--muted)",
                            fontWeight: tab === t ? 600 : 400,
                        }}
                    >
                        {t === "record" ? "Record & Transcribe" : "Text to Speech"}
                        {tab === t && <div className="tab-indicator" />}
                    </button>
                ))}
            </nav>

            {/* Content */}
            <main className="flex-1 px-6 py-8 relative z-10">
                <div className="max-w-2xl mx-auto space-y-6">
                    {tab === "record" && (
                        <>
                            {/* Model selectors */}
                            <ModelSelectors disabled={transcribing || analyzing} />

                            {/* Microphone selector */}
                            {devices.length > 1 && (
                                <div className="flex items-center justify-center gap-2">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--muted)" }}>
                                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                        <line x1="12" x2="12" y1="19" y2="22" />
                                    </svg>
                                    <select
                                        value={selectedId}
                                        onChange={(e) => setSelectedId(e.target.value)}
                                        disabled={transcribing}
                                        className="text-sm rounded-lg px-3 py-1.5 max-w-xs truncate"
                                        style={{
                                            background: "var(--surface-2)",
                                            color: "var(--fg)",
                                            border: "1px solid var(--border)",
                                        }}
                                    >
                                        {devices.map((d) => (
                                            <option key={d.deviceId} value={d.deviceId}>
                                                {d.label}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            )}

                            <Recorder onRecorded={handleRecorded} disabled={transcribing} deviceId={selectedId} />

                            {/* Or upload a file */}
                            <div className="flex items-center gap-3">
                                <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                                <span className="text-xs uppercase tracking-wider" style={{ color: "var(--muted)" }}>or upload a file</span>
                                <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                            </div>

                            <label
                                className="block text-center rounded-lg p-4 cursor-pointer transition-all text-sm hover:border-solid"
                                style={{
                                    border: "1px dashed var(--border)",
                                    color: "var(--muted)",
                                }}
                            >
                                Drop or click to upload audio (WAV, MP3, FLAC, OGG)
                                <input
                                    type="file"
                                    accept="audio/*"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                />
                            </label>

                            {/* Saved recordings library */}
                            <RecordingsLibrary onSelect={handleSelectRecording} refreshKey={refreshRecordings} />

                            {recordingBlob && <AudioPlayer src={recordingBlob} label={recordingLabel} />}

                            {error && (
                                <div className="rounded-lg p-3" style={{ background: "var(--surface)", border: "1px solid var(--error)" }}>
                                    <p className="text-sm" style={{ color: "var(--error)" }}>
                                        {error}
                                    </p>
                                    {error.includes("Docker") && (
                                        <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
                                            Docker containers can take 30-90 seconds to start and load models.
                                            Try again in a moment, or switch to a native engine.
                                        </p>
                                    )}
                                </div>
                            )}

                            <TranscriptionView result={transcription} loading={transcribing} />
                            <AnalysisView result={analysis} loading={analyzing} />
                        </>
                    )}

                    {tab === "tts" && <TtsPlayground />}
                </div>
            </main>

            {/* Footer */}
            <footer className="px-6 py-4 text-center relative z-10">
                <span className="text-xs uppercase tracking-wider" style={{ color: "var(--muted)", opacity: 0.6 }}>
                    All processing runs locally on your machine
                </span>
            </footer>
        </div>
    );
}
