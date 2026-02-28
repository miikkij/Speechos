"use client";

import { useEffect, useState } from "react";
import { getModelOptions, switchModel, type ModelOption } from "@/lib/api";
import { loadSettings, saveSettings } from "@/lib/storage";

interface ModelSelectorsProps {
    disabled?: boolean;
}

export function ModelSelectors({ disabled }: ModelSelectorsProps) {
    const [options, setOptions] = useState<Record<string, ModelOption[]>>({});
    const [current, setCurrent] = useState<Record<string, { engine: string; model: string }>>({});
    const [switching, setSwitching] = useState<string | null>(null);
    const [switchStatus, setSwitchStatus] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        getModelOptions()
            .then(async (data) => {
                setOptions(data.options);
                setCurrent(data.current);

                // Restore saved model selections
                const saved = loadSettings().models;
                if (!saved) return;

                for (const category of Object.keys(saved)) {
                    const savedSel = saved[category];
                    const serverCur = data.current[category];
                    const categoryOpts = data.options[category];
                    if (!savedSel || !categoryOpts) continue;

                    // Check if saved selection is available
                    const isAvailable = categoryOpts.some(
                        (o) => o.engine === savedSel.engine && o.model === savedSel.model && o.installed !== false
                    );
                    if (!isAvailable) continue;

                    // Skip if already the current selection
                    if (serverCur?.engine === savedSel.engine && serverCur?.model === savedSel.model) continue;

                    // Restore saved selection
                    setCurrent((prev) => ({ ...prev, [category]: savedSel }));
                    try {
                        await switchModel(category, savedSel.engine, savedSel.model);
                    } catch {
                        // Restore failed, revert to server's current
                        setCurrent((prev) => ({ ...prev, [category]: serverCur }));
                    }
                }
            })
            .catch(() => { });
    }, []);

    const isDockerEngine = (label: string) => label.includes("[Docker]");

    const handleChange = async (category: string, value: string) => {
        const [engine, model] = value.split("|");
        const opt = options[category]?.find((o) => o.engine === engine && o.model === model);
        const isDocker = opt ? isDockerEngine(opt.label) : false;

        setCurrent((prev) => ({ ...prev, [category]: { engine, model } }));
        setSwitching(category);
        setSwitchStatus(isDocker ? "Starting Docker container..." : null);
        setError(null);

        // Persist to localStorage
        const saved = loadSettings().models ?? {};
        saveSettings({ models: { ...saved, [category]: { engine, model } } });

        try {
            const result = await switchModel(category, engine, model);
            if (result.docker_status === "failed") {
                setError(`Docker container for ${engine} failed to start`);
            } else if (result.docker_status === "ready") {
                setSwitchStatus(null);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : `Failed to switch ${category}`);
        } finally {
            setSwitching(null);
            setSwitchStatus(null);
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
                                <option
                                    key={`${opt.engine}|${opt.model}`}
                                    value={`${opt.engine}|${opt.model}`}
                                    disabled={opt.installed === false}
                                >
                                    {opt.installed === false ? `\u2298 ${opt.label} (not installed)` : opt.label}
                                </option>
                            ))}
                        </select>
                        {switching === key && (
                            <span className="text-xs shrink-0 animate-pulse" style={{ color: "var(--warning)" }}>
                                {switchStatus || "..."}
                            </span>
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
