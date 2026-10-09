"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface AudioControls {
    ref: React.RefObject<HTMLAudioElement | null>;
    src: string | undefined;
    playing: boolean;
    currentTime: number;
    duration: number;
    rate: number;
    toggle: () => void;
    seek: (seconds: number, play?: boolean) => void;
    setRate: (rate: number) => void;
    bind: {
        onPlay: () => void;
        onPause: () => void;
        onEnded: () => void;
        onTimeUpdate: () => void;
        onLoadedMetadata: () => void;
    };
}

/** A hidden <audio> element driven by custom controls. Spread `bind` onto the element and pass `ref`. */
export function useAudioElement(blob: Blob | null): AudioControls {
    const ref = useRef<HTMLAudioElement | null>(null);
    const [src, setSrc] = useState<string | undefined>(undefined);
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [rate, setRateState] = useState(1);

    useEffect(() => {
        if (!blob) {
            setSrc(undefined);
            return;
        }
        const url = URL.createObjectURL(blob);
        setSrc(url);
        setCurrentTime(0);
        setPlaying(false);
        return () => URL.revokeObjectURL(url);
    }, [blob]);

    const toggle = useCallback(() => {
        const el = ref.current;
        if (!el) return;
        if (el.paused) el.play().catch(() => undefined);
        else el.pause();
    }, []);

    const seek = useCallback((seconds: number, play = false) => {
        const el = ref.current;
        if (!el) return;
        el.currentTime = seconds;
        setCurrentTime(seconds);
        if (play) el.play().catch(() => undefined);
    }, []);

    const setRate = useCallback((r: number) => {
        setRateState(r);
        if (ref.current) ref.current.playbackRate = r;
    }, []);

    const bind = {
        onPlay: () => setPlaying(true),
        onPause: () => setPlaying(false),
        onEnded: () => setPlaying(false),
        onTimeUpdate: () => setCurrentTime(ref.current?.currentTime ?? 0),
        onLoadedMetadata: () => {
            const el = ref.current;
            if (!el) return;
            if (Number.isFinite(el.duration)) setDuration(el.duration);
            el.playbackRate = rate;
        },
    };

    return { ref, src, playing, currentTime, duration, rate, toggle, seek, setRate, bind };
}
