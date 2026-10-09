"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type AudioEvent = React.SyntheticEvent<HTMLAudioElement>;

export interface AudioControls {
    /** Callback ref for the <audio> element: `<audio ref={audio.attach} …>`. */
    attach: (el: HTMLAudioElement | null) => void;
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
        onTimeUpdate: (e: AudioEvent) => void;
        onLoadedMetadata: (e: AudioEvent) => void;
    };
}

/** A hidden <audio> element driven by custom controls. Pass `attach` as its ref and spread `bind` onto it. */
export function useAudioElement(blob: Blob | null): AudioControls {
    const elementRef = useRef<HTMLAudioElement | null>(null);
    const [src, setSrc] = useState<{ blob: Blob; url: string } | null>(null);
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [rate, setRateState] = useState(1);

    useEffect(() => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        // Object URLs are an external resource: created here, revoked in cleanup (also under StrictMode re-runs)
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setSrc({ blob, url });
        return () => URL.revokeObjectURL(url);
    }, [blob]);

    const attach = useCallback((node: HTMLAudioElement | null) => {
        elementRef.current = node;
    }, []);

    const toggle = useCallback(() => {
        const el = elementRef.current;
        if (!el) return;
        if (el.paused) el.play().catch(() => undefined);
        else el.pause();
    }, []);

    const seek = useCallback((seconds: number, play = false) => {
        const el = elementRef.current;
        if (!el) return;
        el.currentTime = seconds;
        setCurrentTime(seconds);
        if (play) el.play().catch(() => undefined);
    }, []);

    const setRate = useCallback((r: number) => {
        setRateState(r);
        if (elementRef.current) elementRef.current.playbackRate = r;
    }, []);

    const current = blob && src?.blob === blob ? src.url : undefined;

    return {
        attach,
        src: current,
        playing: current ? playing : false,
        currentTime: current ? currentTime : 0,
        duration: current ? duration : 0,
        rate,
        toggle,
        seek,
        setRate,
        bind: {
            onPlay: () => setPlaying(true),
            onPause: () => setPlaying(false),
            onEnded: () => setPlaying(false),
            onTimeUpdate: (e) => setCurrentTime(e.currentTarget.currentTime),
            onLoadedMetadata: (e) => {
                const a = e.currentTarget;
                setCurrentTime(0);
                if (Number.isFinite(a.duration)) setDuration(a.duration);
                a.playbackRate = rate;
            },
        },
    };
}
