"use client";

import { useEffect, useState } from "react";
import { Panel } from "@/components/ui";
import { getModelOptions, getSystemInfo, switchModel, type ModelOptionsResponse, type SystemInfo } from "@/lib/api";
import { modelDisplayName } from "@/lib/format";

const CATEGORIES: { key: string; title: string; desc: string }[] = [
    { key: "stt", title: "Speech to text", desc: "Turns speech into a transcript with timestamps." },
    { key: "tts", title: "Text to speech", desc: "Default voice for the Synthesize view." },
    { key: "emotion", title: "Emotion", desc: "Recognizes emotion from the voice, not the words." },
    { key: "diarization", title: "Speakers", desc: "Separates who spoke when." },
    { key: "vad", title: "Voice activity", desc: "Finds speech and skips silence." },
];

const COLLAPSED = 4;

function CategoryCard({
    title,
    desc,
    category,
    data,
    onSwitched,
}: {
    title: string;
    desc: string;
    category: string;
    data: ModelOptionsResponse;
    onSwitched: (category: string, engine: string, model: string) => void;
}) {
    const [expanded, setExpanded] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const options = data.options[category] ?? [];
    const cur = data.current[category];
    const finnishCfg = data.current.stt_fi;
    const finnish = finnishCfg && finnishCfg.engine !== "none" ? finnishCfg : null;
    const isActive = (engine: string, model: string) => cur?.engine === engine && cur?.model === model;

    // Active first, then installed, then the rest
    const sorted = [...options].sort((a, b) => {
        const rank = (o: (typeof options)[number]) => (isActive(o.engine, o.model) ? 0 : o.installed === false ? 2 : 1);
        return rank(a) - rank(b);
    });
    const visible = expanded ? sorted : sorted.slice(0, COLLAPSED);

    const choose = async (engine: string, model: string) => {
        setBusy(true);
        setError(null);
        try {
            await switchModel(category, engine, model);
            onSwitched(category, engine, model);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Switching failed");
        } finally {
            setBusy(false);
        }
    };

    if (options.length === 0) return null;

    return (
        <Panel className="min-w-0 gap-2.5">
            <div className="flex flex-col gap-0.5">
                <h2 className="m-0 text-base font-semibold">{title}</h2>
                <span className="text-[13px] text-muted">{desc}</span>
            </div>
            {category === "stt" && finnish && (
                <p className="m-0 rounded-[10px] border border-selected-line bg-selected px-3 py-2.5 text-[13px]">
                    Finnish requests use <span className="font-medium">{modelDisplayName(finnish.model)}</span>: 6.2 % WER vs
                    7.5 % for large-v3 on FLEURS-fi, same speed. Other languages use the model chosen below.
                </p>
            )}
            <fieldset className="m-0 flex flex-col gap-1.5 border-0 p-0" disabled={busy}>
                <legend className="sr-only">{title}</legend>
                {visible.map((o) => {
                    const on = isActive(o.engine, o.model);
                    const missing = o.installed === false;
                    const [name, note] = o.label.replace("[Docker] ", "").split(" (");
                    const status = on ? "Active" : missing ? "Not installed" : o.label.startsWith("[Docker]") ? "Docker" : "Installed";
                    return (
                        <label
                            key={`${o.engine}|${o.model}`}
                            className={`flex items-center gap-3 rounded-[10px] border-[1.5px] p-3 ${
                                on ? "border-accent bg-selected" : "border-line-2 bg-surface-2"
                            } ${missing ? "cursor-not-allowed opacity-75" : "cursor-pointer hover:border-line-3"}`}
                        >
                            <input
                                type="radio"
                                name={category}
                                checked={on}
                                disabled={missing}
                                onChange={() => choose(o.engine, o.model)}
                                className="h-4 w-4 flex-none accent-accent"
                            />
                            <span className="flex min-w-0 flex-auto flex-col">
                                <span className="font-medium">{name}</span>
                                {note && <span className="text-xs text-muted">{note.replace(/\)$/, "")}</span>}
                            </span>
                            <span
                                className={`rounded-full px-2.5 py-0.5 text-xs whitespace-nowrap ${
                                    on
                                        ? "bg-accent font-semibold text-bg"
                                        : missing
                                          ? "border border-dashed border-[#4a505a] text-muted"
                                          : "border border-line-3 text-[#c9cdd3]"
                                }`}
                            >
                                {status}
                            </span>
                        </label>
                    );
                })}
            </fieldset>
            {sorted.length > COLLAPSED && (
                <button type="button" onClick={() => setExpanded(!expanded)} className="self-start text-[13px] text-accent hover:text-accent-hover">
                    {expanded ? "Show fewer" : `Show ${sorted.length - COLLAPSED} more`}
                </button>
            )}
            {error && (
                <p role="alert" className="m-0 text-[13px] text-bad">
                    {error}
                </p>
            )}
        </Panel>
    );
}

export default function ModelsPage() {
    const [info, setInfo] = useState<SystemInfo | null>(null);
    const [data, setData] = useState<ModelOptionsResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        getSystemInfo()
            .then(setInfo)
            .catch(() => setError("Could not reach the API."));
        getModelOptions()
            .then(setData)
            .catch(() => setError("Could not reach the API."));
    }, []);

    const onSwitched = (category: string, engine: string, model: string) => {
        setData((prev) => (prev ? { ...prev, current: { ...prev.current, [category]: { engine, model } } } : prev));
    };

    const hw = info?.hardware;

    return (
        <div className="mx-auto flex max-w-[1400px] flex-col gap-6 px-6 py-8">
            <div>
                <h1 className="m-0 text-[26px] font-semibold tracking-tight">Models &amp; hardware</h1>
                <p className="mt-1.5 mb-0 text-[15px] text-muted">
                    Speechos picks models for your hardware automatically. Change them here; the choice applies to every new analysis.
                </p>
            </div>

            {error && (
                <p role="alert" className="m-0 text-sm text-bad">
                    {error}
                </p>
            )}

            {hw && (
                <section
                    aria-label="Hardware"
                    className="flex flex-wrap items-end gap-x-10 gap-y-5 rounded-xl border border-line bg-surface p-5"
                >
                    <div className="flex flex-col gap-0.5">
                        <span className="text-xs text-muted">Detected tier</span>
                        <span className="font-mono text-xl font-semibold">{hw.tier}</span>
                    </div>
                    <div className="flex flex-col gap-0.5">
                        <span className="text-xs text-muted">{hw.gpu_available ? "GPU" : "Compute"}</span>
                        <span className="text-[15px] font-medium">
                            {hw.gpu_available ? `${hw.gpu_name ?? "GPU"} · ${hw.vram_gb} GB` : `CPU · ${hw.cpu_cores} cores`}
                        </span>
                    </div>
                    <div className="flex flex-col gap-0.5">
                        <span className="text-xs text-muted">Memory</span>
                        <span className="text-[15px] font-medium">{hw.ram_gb} GB RAM</span>
                    </div>
                    <div className="flex flex-col gap-0.5">
                        <span className="text-xs text-muted">Loaded now</span>
                        <span className="text-[15px] font-medium">{info.models.loaded.length} models</span>
                    </div>
                    <p className="m-0 min-w-[240px] flex-1 text-[13px] text-muted">
                        Override the tier with <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-fg">SPEECHOS_TIER</code>,
                        for example <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-fg">gpu-12gb</code>.
                    </p>
                </section>
            )}

            {data && (
                <div className="grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(min(400px, 100%), 1fr))" }}>
                    {CATEGORIES.map((c) => (
                        <CategoryCard key={c.key} title={c.title} desc={c.desc} category={c.key} data={data} onSwitched={onSwitched} />
                    ))}
                </div>
            )}

            <p className="m-0 text-[13px] text-muted">
                Missing engines install as optional extras, for example{" "}
                <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-fg">uv sync --extra gpu</code>. Speaker analysis with
                pyannote needs a Hugging Face token.
            </p>
        </div>
    );
}
