"use client";

import { useEffect, useRef, useState } from "react";
import { loadSettings, saveSettings } from "@/lib/storage";

interface AudioPlayerProps {
    src: string | Blob | null;
    label?: string;
}

export function AudioPlayer({ src, label }: AudioPlayerProps) {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [playing, setPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [objectUrl, setObjectUrl] = useState<string | null>(null);
    const [volume, setVolumeState] = useState(() => {
        return loadSettings().playbackVolume ?? 100;
    });

    useEffect(() => {
        if (src instanceof Blob) {
            const url = URL.createObjectURL(src);
            setObjectUrl(url);
            return () => URL.revokeObjectURL(url);
        }
        setObjectUrl(null);
    }, [src]);

    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.volume = volume / 100;
        }
    }, [volume, src]);

    const audioSrc = objectUrl ?? (typeof src === "string" ? src : undefined);

    const handleVolumeChange = (newVolume: number) => {
        setVolumeState(newVolume);
        if (audioRef.current) {
            audioRef.current.volume = newVolume / 100;
        }
        saveSettings({ playbackVolume: newVolume });
    };

    const togglePlay = () => {
        const audio = audioRef.current;
        if (!audio) return;
        if (playing) {
            audio.pause();
        } else {
            audio.play();
        }
    };

    return (
        <div
            className="flex items-center gap-3 rounded-lg p-3"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
        >
            <button
                onClick={togglePlay}
                disabled={!audioSrc}
                className="w-8 h-8 rounded-full flex items-center justify-center disabled:opacity-40 shrink-0"
                style={{ background: "var(--accent)" }}
            >
                {playing ? "\u23F8" : "\u25B6"}
            </button>

            <div className="flex-1 flex flex-col gap-1 min-w-0">
                {label && (
                    <span className="text-xs" style={{ color: "var(--muted)" }}>
                        {label}
                    </span>
                )}
                <div
                    className="h-1 rounded-full overflow-hidden"
                    style={{ background: "var(--border)" }}
                >
                    <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${progress * 100}%`, background: "var(--accent)" }}
                    />
                </div>
            </div>

            {/* Volume control */}
            <div className="flex items-center gap-1.5 shrink-0">
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                    {volume === 0 ? "\uD83D\uDD07" : volume < 50 ? "\uD83D\uDD09" : "\uD83D\uDD0A"}
                </span>
                <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={volume}
                    onChange={(e) => handleVolumeChange(Number(e.target.value))}
                    className="w-16 h-1 rounded-full appearance-none cursor-pointer"
                    style={{ accentColor: "var(--accent)" }}
                    title={`Volume: ${volume}%`}
                />
            </div>

            {audioSrc && (
                <audio
                    ref={audioRef}
                    src={audioSrc}
                    onPlay={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onEnded={() => {
                        setPlaying(false);
                        setProgress(0);
                    }}
                    onTimeUpdate={(e) => {
                        const audio = e.currentTarget;
                        if (audio.duration) setProgress(audio.currentTime / audio.duration);
                    }}
                />
            )}
        </div>
    );
}
