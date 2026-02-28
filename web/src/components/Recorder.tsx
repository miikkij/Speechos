"use client";

import { useState } from "react";
import { useRecorder, type AudioSettings, DEFAULT_AUDIO_SETTINGS } from "@/hooks/useRecorder";
import { loadSettings, saveSettings } from "@/lib/storage";

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
    const [settings, setSettingsRaw] = useState<AudioSettings>(() => {
        const saved = loadSettings().audioSettings;
        return saved ?? DEFAULT_AUDIO_SETTINGS;
    });
    const [showSettings, setShowSettings] = useState(false);
    const { isRecording, duration, start, stop, audioLevel } = useRecorder(deviceId, settings);

    const setSettings = (updater: AudioSettings | ((prev: AudioSettings) => AudioSettings)) => {
        setSettingsRaw((prev) => {
            const next = typeof updater === "function" ? updater(prev) : updater;
            saveSettings({ audioSettings: next });
            return next;
        });
    };

    const handleClick = async () => {
        if (isRecording) {
            const blob = await stop();
            onRecorded(blob);
        } else {
            await start();
        }
    };

    const handleReset = () => {
        setSettingsRaw(DEFAULT_AUDIO_SETTINGS);
        saveSettings({ audioSettings: undefined });
    };

    const gainPercent = Math.round(settings.gain * 100);

    return (
        <div className="flex flex-col items-center gap-4">
            {/* Mic button with pulse rings */}
            <div className="relative w-24 h-24 flex items-center justify-center">
                {/* Spotlight glow behind mic */}
                {isRecording && (
                    <div
                        className="absolute w-48 h-48 rounded-full pointer-events-none"
                        style={{
                            background: `radial-gradient(circle, rgba(232, 145, 58, ${0.08 + audioLevel * 0.12}) 0%, transparent 70%)`,
                        }}
                    />
                )}

                {/* Pulse rings */}
                {isRecording && (
                    <>
                        <div className="pulse-ring" />
                        <div className="pulse-ring pulse-ring-2" />
                        <div className="pulse-ring pulse-ring-3" />
                    </>
                )}

                <button
                    onClick={handleClick}
                    disabled={disabled}
                    className="relative w-20 h-20 rounded-full border-2 transition-all duration-300 flex items-center justify-center disabled:opacity-50 z-10"
                    style={{
                        borderColor: isRecording ? "var(--accent)" : "var(--border)",
                        background: isRecording
                            ? `rgba(232, 145, 58, ${0.1 + audioLevel * 0.25})`
                            : "var(--surface)",
                        animation: isRecording ? "glow-breathe 2s ease-in-out infinite" : "none",
                    }}
                >
                    {isRecording ? (
                        <div className="w-6 h-6 rounded-sm" style={{ background: "var(--accent)" }} />
                    ) : (
                        <div className="w-8 h-8 rounded-full" style={{ background: "var(--accent)", opacity: 0.9 }} />
                    )}
                </button>
            </div>

            {/* Timer / instructions */}
            <span
                className="text-sm transition-colors"
                style={{ color: isRecording ? "var(--accent)" : "var(--muted)" }}
            >
                {isRecording ? formatTime(duration) : "Click to record"}
            </span>

            {/* Equalizer bars */}
            {isRecording && (
                <div
                    className="flex items-end justify-center gap-[3px] h-7"
                    style={{ opacity: 0.4 + audioLevel * 0.6 }}
                >
                    <div className="eq-bar eq-bar-1" style={{ background: "var(--accent)" }} />
                    <div className="eq-bar eq-bar-2" style={{ background: "var(--accent)" }} />
                    <div className="eq-bar eq-bar-3" style={{ background: "var(--accent)" }} />
                    <div className="eq-bar eq-bar-4" style={{ background: "var(--accent)" }} />
                    <div className="eq-bar eq-bar-5" style={{ background: "var(--accent)" }} />
                </div>
            )}

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
                    {/* Sample Rate */}
                    <div className="space-y-1">
                        <div className="flex items-center justify-between">
                            <label className="text-xs" style={{ color: "var(--muted)" }}>
                                Sample Rate
                            </label>
                            <span className="text-xs font-mono" style={{ color: "var(--fg)" }}>
                                {(settings.sampleRate / 1000).toFixed(settings.sampleRate % 1000 === 0 ? 0 : 1)} kHz
                            </span>
                        </div>
                        <div className="flex gap-1">
                            {([
                                { rate: 16000, label: "16k" },
                                { rate: 44100, label: "44.1k" },
                                { rate: 48000, label: "48k" },
                            ] as const).map(({ rate, label }) => (
                                <button
                                    key={rate}
                                    onClick={() => setSettings((s) => ({ ...s, sampleRate: rate }))}
                                    disabled={isRecording}
                                    className="flex-1 text-xs py-1 rounded transition-colors disabled:opacity-50"
                                    style={{
                                        background: settings.sampleRate === rate ? "var(--accent)" : "var(--surface-2)",
                                        color: settings.sampleRate === rate ? "#fff" : "var(--muted)",
                                        border: "1px solid var(--border)",
                                    }}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>

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
                            className="w-full"
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
                        onClick={handleReset}
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
