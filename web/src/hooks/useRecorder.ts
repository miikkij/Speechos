"use client";

import { useCallback, useRef, useState } from "react";

export interface AudioSettings {
    gain: number;             // 0.0 – 3.0 (1.0 = normal)
    sampleRate: number;       // 16000, 44100, or 48000
    noiseSuppression: boolean;
    echoCancellation: boolean;
    autoGainControl: boolean;
}

export const DEFAULT_AUDIO_SETTINGS: AudioSettings = {
    gain: 1.0,
    sampleRate: 48000,
    noiseSuppression: true,
    echoCancellation: true,
    autoGainControl: true,
};

export interface UseRecorderReturn {
    isRecording: boolean;
    duration: number;
    start: () => Promise<void>;
    stop: () => Promise<Blob>;
    audioLevel: number;
}

export function useRecorder(deviceId?: string, settings?: AudioSettings): UseRecorderReturn {
    const [isRecording, setIsRecording] = useState(false);
    const [duration, setDuration] = useState(0);
    const [audioLevel, setAudioLevel] = useState(0);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const gainNodeRef = useRef<GainNode | null>(null);
    const animFrameRef = useRef<number>(0);
    const resolveStopRef = useRef<((blob: Blob) => void) | null>(null);

    const s = settings ?? DEFAULT_AUDIO_SETTINGS;

    const start = useCallback(async () => {
        const constraints: MediaStreamConstraints = {
            audio: {
                channelCount: 1,
                sampleRate: s.sampleRate,
                noiseSuppression: s.noiseSuppression,
                echoCancellation: s.echoCancellation,
                autoGainControl: s.autoGainControl,
                ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
            },
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        // Audio processing chain: source → gain → analyser → destination
        const audioCtx = new AudioContext({ sampleRate: s.sampleRate });
        const source = audioCtx.createMediaStreamSource(stream);

        // Gain node for amplification
        const gainNode = audioCtx.createGain();
        gainNode.gain.value = s.gain;
        gainNodeRef.current = gainNode;
        source.connect(gainNode);

        // Analyser for level meter (reads post-gain signal)
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        gainNode.connect(analyser);
        analyserRef.current = analyser;

        // Route processed audio to a MediaStream for recording
        const dest = audioCtx.createMediaStreamDestination();
        gainNode.connect(dest);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const updateLevel = () => {
            analyser.getByteFrequencyData(dataArray);
            const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
            setAudioLevel(avg / 255);
            animFrameRef.current = requestAnimationFrame(updateLevel);
        };
        updateLevel();

        // Record from the gain-processed stream
        const mr = new MediaRecorder(dest.stream, { mimeType: "audio/webm;codecs=opus" });
        chunksRef.current = [];

        mr.ondataavailable = (e) => {
            if (e.data.size > 0) chunksRef.current.push(e.data);
        };

        mr.onstop = () => {
            stream.getTracks().forEach((t) => t.stop());
            audioCtx.close();
            cancelAnimationFrame(animFrameRef.current);
            setAudioLevel(0);

            const blob = new Blob(chunksRef.current, { type: "audio/webm" });
            resolveStopRef.current?.(blob);
        };

        mediaRecorderRef.current = mr;
        mr.start(250);
        setIsRecording(true);
        setDuration(0);

        const t0 = Date.now();
        timerRef.current = setInterval(() => {
            setDuration(Math.floor((Date.now() - t0) / 1000));
        }, 200);
    }, [deviceId, s.gain, s.sampleRate, s.noiseSuppression, s.echoCancellation, s.autoGainControl]);

    const stop = useCallback((): Promise<Blob> => {
        return new Promise((resolve) => {
            resolveStopRef.current = resolve;
            if (timerRef.current) clearInterval(timerRef.current);
            setIsRecording(false);
            mediaRecorderRef.current?.stop();
        });
    }, []);

    return { isRecording, duration, start, stop, audioLevel };
}
