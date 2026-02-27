"use client";

import { useEffect, useState } from "react";
import { getModelOptions, switchModel, type ModelOption } from "@/lib/api";

interface ModelSelectorsProps {
    disabled?: boolean;
}

export function ModelSelectors({ disabled }: ModelSelectorsProps) {
    const [options, setOptions] = useState<Record<string, ModelOption[]>>({});
    const [current, setCurrent] = useState<Record<string, { engine: string; model: string }>>({});
    const [switching, setSwitching] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        getModelOptions()
            .then((data) => {
                setOptions(data.options);
                setCurrent(data.current);
            })
            .catch(() => {});
    }, []);

    const handleChange = async (category: string, value: string) => {
        const [engine, model] = value.split("|");
        setCurrent((prev) => ({ ...prev, [category]: { engine, model } }));
        setSwitching(category);
        setError(null);
        try {
            await switchModel(category, engine, model);
        } catch (err) {
            setError(err instanceof Error ? err.message : `Failed to switch ${category}`);
        } finally {
            setSwitching(null);
        }
    };

    const categories: { key: string; label: string }[] = [
        { key: "stt", label: "Speech-to-Text" },
        { key: "emotion", label: "Emotion" },
        { key: "diarization", label: "Speaker Analysis" },
        { key: "vad", label: "Voice Activity" },
    ];

    if (Object.keys(options).length === 0) return null;

    return (
        <div
            className="rounded-lg p-4 space-y-3"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
            <h3 className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                Analysis Models
            </h3>

            {categories.map(({ key, label }) => {
                const opts = options[key];
                if (!opts) return null;
                const cur = current[key];
                const curValue = cur ? `${cur.engine}|${cur.model}` : "";

                return (
                    <div key={key} className="flex items-center gap-2">
                        <label className="text-xs w-28 shrink-0" style={{ color: "var(--muted)" }}>
                            {label}
                        </label>
                        <select
                            value={curValue}
                            onChange={(e) => handleChange(key, e.target.value)}
                            disabled={disabled || switching === key}
                            className="flex-1 text-xs rounded-lg px-2 py-1.5 truncate"
                            style={{
                                background: "var(--surface-2)",
                                color: "var(--fg)",
                                border: "1px solid var(--border)",
                            }}
                        >
                            {opts.map((opt) => (
                                <option key={`${opt.engine}|${opt.model}`} value={`${opt.engine}|${opt.model}`}>
                                    {opt.label}
                                </option>
                            ))}
                        </select>
                        {switching === key && (
                            <span className="text-xs shrink-0" style={{ color: "var(--warning)" }}>...</span>
                        )}
                    </div>
                );
            })}

            {error && (
                <p className="text-xs" style={{ color: "var(--error)" }}>
                    {error}
                </p>
            )}
        </div>
    );
}
