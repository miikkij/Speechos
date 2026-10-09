"use client";

import { formatTime } from "@/lib/format";

interface WaveformProps {
    values: number[] | null;
    duration: number;
    currentTime: number;
    onSeek?: (seconds: number) => void;
    height?: number;
    /** Optional mask: true where there is speech; silent buckets draw as a flat line. */
    speech?: boolean[];
}

/** Bars of peak amplitude; the played part is drawn in the accent color. A transparent range input on top seeks. */
export function Waveform({ values, duration, currentTime, onSeek, height = 72, speech }: WaveformProps) {
    const bars = values ?? Array<number>(140).fill(0.08);
    const progress = duration > 0 ? currentTime / duration : 0;

    return (
        <div className="relative flex min-w-0 items-center gap-px sm:gap-[2px]" style={{ height }}>
            {bars.map((v, i) => {
                const played = (i + 0.5) / bars.length <= progress;
                const silent = speech ? !speech[i] : false;
                const h = silent ? 4 : Math.max(4, Math.round(v * (height - 12)));
                return (
                    <span
                        key={i}
                        aria-hidden="true"
                        className={`min-w-0 flex-1 rounded-sm ${played ? "bg-accent" : "bg-idle-bar"}`}
                        style={{ height: h }}
                    />
                );
            })}
            {duration > 0 && (
                <span
                    aria-hidden="true"
                    className="pointer-events-none absolute -top-1 -bottom-1 w-0.5 rounded-sm bg-fg"
                    style={{ left: `${progress * 100}%` }}
                />
            )}
            {onSeek && duration > 0 && (
                <input
                    type="range"
                    min={0}
                    max={duration}
                    step={0.1}
                    value={currentTime}
                    onChange={(e) => onSeek(Number(e.target.value))}
                    aria-label="Seek"
                    aria-valuetext={`${formatTime(currentTime)} of ${formatTime(duration)}`}
                    className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                />
            )}
        </div>
    );
}
