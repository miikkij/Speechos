"use client";

import { useRecorder } from "@/hooks/useRecorder";

function formatTime(s: number): string {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
}

interface RecorderProps {
    onRecorded: (blob: Blob) => void;
    disabled?: boolean;
}

export function Recorder({ onRecorded, disabled }: RecorderProps) {
    const { isRecording, duration, start, stop, audioLevel } = useRecorder();

    const handleClick = async () => {
        if (isRecording) {
            const blob = await stop();
            onRecorded(blob);
        } else {
            await start();
        }
    };

    return (
        <div className="flex flex-col items-center gap-4">
            <button
                onClick={handleClick}
                disabled={disabled}
                className="relative w-20 h-20 rounded-full border-2 transition-all duration-200 flex items-center justify-center disabled:opacity-50"
                style={{
                    borderColor: isRecording ? "var(--error)" : "var(--accent)",
                    background: isRecording
                        ? `rgba(239, 68, 68, ${0.1 + audioLevel * 0.3})`
                        : "var(--surface)",
                    boxShadow: isRecording
                        ? `0 0 ${20 + audioLevel * 40}px rgba(239, 68, 68, ${0.3 + audioLevel * 0.4})`
                        : "none",
                }}
            >
                {isRecording ? (
                    <div className="w-6 h-6 rounded-sm" style={{ background: "var(--error)" }} />
                ) : (
                    <div className="w-8 h-8 rounded-full" style={{ background: "var(--accent)" }} />
                )}
            </button>

            <span className="text-sm" style={{ color: isRecording ? "var(--error)" : "var(--muted)" }}>
                {isRecording ? formatTime(duration) : "Click to record"}
            </span>
        </div>
    );
}
