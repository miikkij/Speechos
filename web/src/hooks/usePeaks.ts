"use client";

import { useEffect, useState } from "react";

export interface Peaks {
    /** Normalized 0–1 amplitude per bucket. */
    values: number[];
    duration: number;
    sampleRate: number;
}

/** Decode an audio blob in the browser and reduce it to `buckets` peak values for a waveform. */
export function usePeaks(blob: Blob | null, buckets: number): Peaks | null {
    const [peaks, setPeaks] = useState<Peaks | null>(null);

    useEffect(() => {
        if (!blob) {
            setPeaks(null);
            return;
        }
        let cancelled = false;
        const ctx = new AudioContext();
        blob
            .arrayBuffer()
            .then((buf) => ctx.decodeAudioData(buf))
            .then((audio) => {
                if (cancelled) return;
                const data = audio.getChannelData(0);
                const size = Math.max(1, Math.floor(data.length / buckets));
                const values: number[] = [];
                let max = 0;
                for (let i = 0; i < buckets; i++) {
                    let peak = 0;
                    const end = Math.min(data.length, (i + 1) * size);
                    for (let j = i * size; j < end; j++) {
                        const v = Math.abs(data[j]);
                        if (v > peak) peak = v;
                    }
                    values.push(peak);
                    if (peak > max) max = peak;
                }
                setPeaks({
                    values: values.map((v) => (max > 0 ? v / max : 0)),
                    duration: audio.duration,
                    sampleRate: audio.sampleRate,
                });
            })
            .catch(() => {
                if (!cancelled) setPeaks(null);
            })
            .finally(() => {
                ctx.close().catch(() => undefined);
            });
        return () => {
            cancelled = true;
        };
    }, [blob, buckets]);

    return peaks;
}
