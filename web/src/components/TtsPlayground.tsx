"use client";

import { useEffect, useState } from "react";
import { synthesize, getModelOptions, switchModel, type ModelOption } from "@/lib/api";
import { AudioPlayer } from "./AudioPlayer";

export function TtsPlayground() {
    const [text, setText] = useState("");
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [loading, setLoading] = useState(false);
    const [switching, setSwitching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [ttsOptions, setTtsOptions] = useState<ModelOption[]>([]);
    const [selectedVoice, setSelectedVoice] = useState("");

    useEffect(() => {
        getModelOptions()
            .then((data) => {
                setTtsOptions(data.options.tts || []);
                const cur = data.current.tts;
                if (cur) {
                    setSelectedVoice(`${cur.engine}|${cur.model}`);
                }
            })
            .catch(() => {});
    }, []);

    const handleVoiceChange = async (value: string) => {
        setSelectedVoice(value);
        const [engine, model] = value.split("|");
        setSwitching(true);
        setError(null);
        try {
            await switchModel("tts", engine, model);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to switch voice");
        } finally {
            setSwitching(false);
        }
    };

    const handleSynthesize = async () => {
        if (!text.trim()) return;
        setLoading(true);
        setError(null);
        try {
            const blob = await synthesize(text);
            setAudioBlob(blob);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Synthesis failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            <h2 className="text-lg font-semibold">Text to Speech</h2>

            {/* Voice selector */}
            {ttsOptions.length > 0 && (
                <div className="flex items-center gap-2">
                    <label className="text-xs shrink-0" style={{ color: "var(--muted)" }}>
                        Voice
                    </label>
                    <select
                        value={selectedVoice}
                        onChange={(e) => handleVoiceChange(e.target.value)}
                        disabled={switching || loading}
                        className="flex-1 text-sm rounded-lg px-3 py-1.5 truncate"
                        style={{
                            background: "var(--surface-2)",
                            color: "var(--fg)",
                            border: "1px solid var(--border)",
                        }}
                    >
                        {ttsOptions.map((opt) => (
                            <option key={`${opt.engine}|${opt.model}`} value={`${opt.engine}|${opt.model}`}>
                                {opt.label}
                            </option>
                        ))}
                    </select>
                    {switching && (
                        <span className="text-xs" style={{ color: "var(--warning)" }}>Switching...</span>
                    )}
                </div>
            )}

            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text to synthesize..."
                rows={3}
                maxLength={10000}
                className="w-full rounded-lg p-3 text-sm resize-y outline-none focus:ring-1"
                style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    color: "var(--fg)",
                }}
            />

            <div className="flex items-center gap-3">
                <button
                    onClick={handleSynthesize}
                    disabled={loading || !text.trim() || switching}
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                    style={{ background: "var(--accent)", color: "#fff" }}
                >
                    {loading ? "Synthesizing..." : "Speak"}
                </button>
                <span className="text-xs" style={{ color: "var(--muted)" }}>
                    {text.length}/10000
                </span>
            </div>

            {error && (
                <p className="text-sm" style={{ color: "var(--error)" }}>
                    {error}
                </p>
            )}

            {audioBlob && <AudioPlayer src={audioBlob} label="Synthesized speech" />}
        </div>
    );
}
