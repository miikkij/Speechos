"use client";

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { useLanguage } from "@/hooks/useLanguage";
import {
    analyze,
    getRecordingBlob,
    transcribe,
    type AnalysisResult,
    type Language,
    type TranscriptionResult,
} from "@/lib/api";

export interface AnalysisOptions {
    transcript: boolean;
    voice: boolean;
}

export interface RecordingResult {
    blob: Blob | null;
    transcription: TranscriptionResult | null;
    analysis: AnalysisResult | null;
    transcribing: boolean;
    analyzing: boolean;
    error: string | null;
}

export interface SynthesisClip {
    id: number;
    text: string;
    voice: string;
    blob: Blob;
}

interface AppState {
    language: Language;
    setLanguage: (lang: Language) => void;
    options: AnalysisOptions;
    setOptions: (opts: AnalysisOptions) => void;
    /** Bumped whenever the recordings list on the server changes. */
    recordingsVersion: number;
    bumpRecordings: () => void;
    results: Record<string, RecordingResult>;
    runAnalysis: (recordingId: string, force?: boolean) => void;
    clips: SynthesisClip[];
    addClip: (clip: Omit<SynthesisClip, "id">) => void;
}

const AppContext = createContext<AppState | null>(null);

const EMPTY: RecordingResult = {
    blob: null,
    transcription: null,
    analysis: null,
    transcribing: false,
    analyzing: false,
    error: null,
};

export function AppProvider({ children }: { children: React.ReactNode }) {
    const [language, setLanguage] = useLanguage();
    const [options, setOptions] = useState<AnalysisOptions>({ transcript: true, voice: true });
    const [recordingsVersion, setRecordingsVersion] = useState(0);
    const [results, setResults] = useState<Record<string, RecordingResult>>({});
    const [clips, setClips] = useState<SynthesisClip[]>([]);
    const started = useRef(new Set<string>());
    const clipId = useRef(0);

    const patch = useCallback((id: string, update: Partial<RecordingResult>) => {
        setResults((prev) => ({ ...prev, [id]: { ...(prev[id] ?? EMPTY), ...update } }));
    }, []);

    const runAnalysis = useCallback(
        (recordingId: string, force = false) => {
            if (started.current.has(recordingId) && !force) return;
            started.current.add(recordingId);
            const wantTranscript = options.transcript;
            const wantVoice = options.voice;
            patch(recordingId, {
                ...EMPTY,
                transcribing: wantTranscript,
                analyzing: wantVoice,
            });

            (async () => {
                let blob: Blob;
                try {
                    blob = await getRecordingBlob(recordingId);
                    patch(recordingId, { blob });
                } catch (err) {
                    patch(recordingId, {
                        transcribing: false,
                        analyzing: false,
                        error: err instanceof Error ? err.message : "Could not load the recording",
                    });
                    return;
                }
                const jobs: Promise<void>[] = [];
                if (wantTranscript) {
                    jobs.push(
                        transcribe(blob, language)
                            .then((transcription) => patch(recordingId, { transcription, transcribing: false }))
                            .catch((err: unknown) =>
                                patch(recordingId, {
                                    transcribing: false,
                                    error: err instanceof Error ? err.message : "Transcription failed",
                                }),
                            ),
                    );
                }
                if (wantVoice) {
                    jobs.push(
                        analyze(blob)
                            .then((analysis) => patch(recordingId, { analysis, analyzing: false }))
                            .catch(() => patch(recordingId, { analyzing: false })),
                    );
                }
                await Promise.all(jobs);
            })();
        },
        [language, options, patch],
    );

    const addClip = useCallback((clip: Omit<SynthesisClip, "id">) => {
        clipId.current += 1;
        const id = clipId.current;
        setClips((prev) => [{ ...clip, id }, ...prev].slice(0, 10));
    }, []);

    const value = useMemo<AppState>(
        () => ({
            language,
            setLanguage,
            options,
            setOptions,
            recordingsVersion,
            bumpRecordings: () => setRecordingsVersion((n) => n + 1),
            results,
            runAnalysis,
            clips,
            addClip,
        }),
        [language, setLanguage, options, recordingsVersion, results, runAnalysis, clips, addClip],
    );

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppState {
    const ctx = useContext(AppContext);
    if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
    return ctx;
}
