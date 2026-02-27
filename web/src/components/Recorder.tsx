"use client";

import { useState } from "react";
import { useRecorder, type AudioSettings, DEFAULT_AUDIO_SETTINGS } from "@/hooks/useRecorder";

function formatTime(s: number): string {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
}

interface RecorderProps {
    onRecorded: (blob: Blob) => void;
    disabled?: boolean;
    deviceId?: string;
}

export function Recorder({ onRecorded, disabled, deviceId }: RecorderProps) {
    const [settings, setSettings] = useState<AudioSettings>(DEFAULT_AUDIO_SETTINGS);
    const [showSettings, setShowSettings] = useState(false);
    const { isRecording, duration, start, stop, audioLevel } = useRecorder(deviceId, settings);

    const handleClick = async () => {
        if (isRecording) {
            const blob = await stop();
            onRecorded(blob);
        } else {
            await start();
        }
    };

    const gainPercent = Math.round(settings.gain * 100);

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

            {/* Settings toggle */}
            <button
                onClick={() => setShowSettings(!showSettings)}
                className="text-xs px-3 py-1 rounded-lg transition-colors"
                style={{
                    color: "var(--muted)",
                    background: showSettings ? "var(--surface-2)" : "transparent",
                    border: "1px solid var(--border)",
                }}
            >
                {showSettings ? "Hide settings" : "Mic settings"}
            </button>

            {/* Audio settings panel */}
            {showSettings && (
                <div
                    className="w-full max-w-xs rounded-lg p-4 space-y-3"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                >
                    {/* Gain / Amplification */}
                    <div className="space-y-1">
                        <div className="flex items-center justify-between">
                            <label className="text-xs" style={{ color: "var(--muted)" }}>
                                Gain
                            </label>
                            <span className="text-xs font-mono" style={{ color: "var(--fg)" }}>
                                {gainPercent}%
                            </span>
                        </div>
                        <input
                            type="range"
                            min={0}
                            max={300}
                            step={10}
                            value={gainPercent}
                            onChange={(e) =>
                                setSettings((s) => ({ ...s, gain: Number(e.target.value) / 100 }))
                            }
                            disabled={isRecording}
                            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                            style={{ accentColor: "var(--accent)" }}
                        />
                        <div className="flex justify-between text-xs" style={{ color: "var(--muted)" }}>
                            <span>Mute</span>
                            <span>Normal</span>
                            <span>3x</span>
                        </div>
                    </div>

                    {/* Toggle switches */}
                    <div className="space-y-2 pt-1">
                        <ToggleRow
                            label="Noise Suppression"
                            checked={settings.noiseSuppression}
                            disabled={isRecording}
                            onChange={(v) => setSettings((s) => ({ ...s, noiseSuppression: v }))}
                        />
                        <ToggleRow
                            label="Echo Cancellation"
                            checked={settings.echoCancellation}
                            disabled={isRecording}
                            onChange={(v) => setSettings((s) => ({ ...s, echoCancellation: v }))}
                        />
                        <ToggleRow
                            label="Auto Gain Control"
                            checked={settings.autoGainControl}
                            disabled={isRecording}
                            onChange={(v) => setSettings((s) => ({ ...s, autoGainControl: v }))}
                        />
                    </div>

                    {/* Reset */}
                    <button
                        onClick={() => setSettings(DEFAULT_AUDIO_SETTINGS)}
                        disabled={isRecording}
                        className="w-full text-xs py-1.5 rounded-lg transition-colors disabled:opacity-50"
                        style={{
                            color: "var(--muted)",
                            border: "1px solid var(--border)",
                            background: "transparent",
                        }}
                    >
                        Reset to defaults
                    </button>
                </div>
            )}
        </div>
    );
}

function ToggleRow({
    label,
    checked,
    disabled,
    onChange,
}: {
    label: string;
    checked: boolean;
    disabled?: boolean;
    onChange: (v: boolean) => void;
}) {
    return (
        <label
            className="flex items-center justify-between cursor-pointer"
            style={{ opacity: disabled ? 0.5 : 1 }}
        >
            <span className="text-xs" style={{ color: "var(--muted)" }}>
                {label}
            </span>
            <button
                type="button"
                role="switch"
                aria-checked={checked}
                disabled={disabled}
                onClick={() => onChange(!checked)}
                className="relative w-8 h-4 rounded-full transition-colors"
                style={{
                    background: checked ? "var(--accent)" : "var(--border)",
                }}
            >
                <span
                    className="absolute top-0.5 w-3 h-3 rounded-full transition-transform"
                    style={{
                        background: "#fff",
                        left: checked ? "calc(100% - 14px)" : "2px",
                    }}
                />
            </button>
        </label>
    );
}
