"use client";

import { useCallback, useState } from "react";
import { Recorder } from "@/components/Recorder";
import { AudioPlayer } from "@/components/AudioPlayer";
import { TranscriptionView } from "@/components/TranscriptionView";
import { AnalysisView } from "@/components/AnalysisView";
import { TtsPlayground } from "@/components/TtsPlayground";
import { StatusBar } from "@/components/StatusBar";
import { transcribe, analyze, uploadRecording, type TranscriptionResult, type AnalysisResult } from "@/lib/api";

type Tab = "record" | "tts";

export default function Home() {
    const [tab, setTab] = useState<Tab>("record");
    const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
    const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
    const [transcribing, setTranscribing] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleRecorded = useCallback(async (blob: Blob) => {
        setRecordingBlob(blob);
        setTranscription(null);
        setAnalysis(null);
        setError(null);
        setTranscribing(true);
        setAnalyzing(true);
        try {
            const [transcribeResult] = await Promise.all([
                transcribe(blob),
                uploadRecording(blob).catch(() => null),
            ]);
            setTranscription(transcribeResult);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Transcription failed");
        } finally {
            setTranscribing(false);
        }
        try {
            const analysisResult = await analyze(blob);
            setAnalysis(analysisResult);
        } catch (err) {
            // Analysis failure is non-critical — transcription still shows
        } finally {
            setAnalyzing(false);
        }
    }, []);

    const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setRecordingBlob(file);
        setTranscription(null);
        setAnalysis(null);
        setError(null);
        setTranscribing(true);
        setAnalyzing(true);
        try {
            const result = await transcribe(file);
            setTranscription(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Transcription failed");
        } finally {
            setTranscribing(false);
        }
        try {
            const analysisResult = await analyze(file);
            setAnalysis(analysisResult);
        } catch (err) {
            // Analysis failure is non-critical
        } finally {
            setAnalyzing(false);
        }
    }, []);

    return (
        <div className="flex flex-col min-h-screen">
            <StatusBar />

            {/* Header */}
            <header className="px-6 py-6">
                <h1 className="text-2xl font-bold tracking-tight">Speechos</h1>
                <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
                    Local-first speech analysis &amp; synthesis
                </p>
            </header>

            {/* Tabs */}
            <nav className="px-6 flex gap-1">
                {(["record", "tts"] as const).map((t) => (
                    <button
                        key={t}
                        onClick={() => setTab(t)}
                        className="px-4 py-2 rounded-t-lg text-sm font-medium transition-colors"
                        style={{
                            background: tab === t ? "var(--surface)" : "transparent",
                            color: tab === t ? "var(--fg)" : "var(--muted)",
                            borderBottom: tab === t ? "2px solid var(--accent)" : "2px solid transparent",
                        }}
                    >
                        {t === "record" ? "Record & Transcribe" : "Text to Speech"}
                    </button>
                ))}
            </nav>

            {/* Content */}
            <main className="flex-1 px-6 py-6">
                <div className="max-w-2xl mx-auto space-y-6">
                    {tab === "record" && (
                        <>
                            <Recorder onRecorded={handleRecorded} disabled={transcribing} />

                            {/* Or upload a file */}
                            <div className="flex items-center gap-3">
                                <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                                <span className="text-xs" style={{ color: "var(--muted)" }}>or upload a file</span>
                                <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                            </div>

                            <label
                                className="block text-center rounded-lg p-4 cursor-pointer transition-colors text-sm"
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

                            {recordingBlob && <AudioPlayer src={recordingBlob} label="Recording" />}

                            {error && (
                                <p className="text-sm" style={{ color: "var(--error)" }}>
                                    {error}
                                </p>
                            )}

                            <TranscriptionView result={transcription} loading={transcribing} />
                            <AnalysisView result={analysis} loading={analyzing} />
                        </>
                    )}

                    {tab === "tts" && <TtsPlayground />}
                </div>
            </main>

            {/* Footer */}
            <footer className="px-6 py-3 text-center">
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                    All processing runs locally on your machine
                </span>
            </footer>
        </div>
    );
}
