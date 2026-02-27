"use client";

import { useCallback, useRef, useState } from "react";

export interface UseRecorderReturn {
    isRecording: boolean;
    duration: number;
    start: () => Promise<void>;
    stop: () => Promise<Blob>;
    audioLevel: number;
}

export function useRecorder(): UseRecorderReturn {
    const [isRecording, setIsRecording] = useState(false);
    const [duration, setDuration] = useState(0);
    const [audioLevel, setAudioLevel] = useState(0);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animFrameRef = useRef<number>(0);
    const resolveStopRef = useRef<((blob: Blob) => void) | null>(null);

    const start = useCallback(async () => {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000 },
        });

        // Audio level meter
        const audioCtx = new AudioContext();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        analyserRef.current = analyser;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const updateLevel = () => {
            analyser.getByteFrequencyData(dataArray);
            const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
            setAudioLevel(avg / 255);
            animFrameRef.current = requestAnimationFrame(updateLevel);
        };
        updateLevel();

        const mr = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
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
        mr.start(250); // collect chunks every 250ms
        setIsRecording(true);
        setDuration(0);

        const t0 = Date.now();
        timerRef.current = setInterval(() => {
            setDuration(Math.floor((Date.now() - t0) / 1000));
        }, 200);
    }, []);

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
