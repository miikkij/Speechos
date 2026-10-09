"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { PauseIcon, PlayIcon, SpeakerIcon } from "@/components/Icons";
import { Eyebrow, Panel, buttonPrimary, buttonSecondary } from "@/components/ui";
import { Waveform } from "@/components/Waveform";
import { useAudioElement } from "@/hooks/useAudioElement";
import { usePeaks } from "@/hooks/usePeaks";
import { getModelOptions, switchModel, synthesize } from "@/lib/api";
import { formatTime, wavSampleRate } from "@/lib/format";
import { useApp } from "@/lib/store";
import { speaksLanguage, toVoices, type Voice } from "@/lib/voices";

const MAX_CHARS = 10000;
const LANGUAGE_NAMES: Record<string, string> = { fi: "Finnish", en: "English", auto: "All" };

function ClipRow({ text, voice, blob }: { text: string; voice: string; blob: Blob }) {
    const { attach: attachAudio, ...audio } = useAudioElement(blob);
    return (
        <div className="flex flex-wrap items-center gap-3 rounded-[10px] border border-line bg-surface px-3.5 py-2.5">
            <button
                type="button"
                onClick={audio.toggle}
                aria-label={audio.playing ? "Pause clip" : "Play clip"}
                className="flex h-9 w-9 flex-none items-center justify-center rounded-full border border-line-2 bg-surface-2 text-fg"
            >
                {audio.playing ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
            </button>
            <span className="min-w-0 flex-[1_1_240px] truncate">{text}</span>
            <span className="text-xs text-muted">{voice}</span>
            <audio ref={attachAudio} src={audio.src} {...audio.bind} />
        </div>
    );
}

export default function SynthesizePage() {
    const { language, clips, addClip } = useApp();
    const [text, setText] = useState("");
    const [speed, setSpeed] = useState(1);
    const [voices, setVoices] = useState<Voice[]>([]);
    const [current, setCurrent] = useState<string>("");
    const [showAll, setShowAll] = useState(false);
    const [switching, setSwitching] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<{ blob: Blob; voice: string; seconds: number; sampleRate: number | null } | null>(null);

    const peaks = usePeaks(result?.blob ?? null, 90);
    const { attach: attachAudio, ...audio } = useAudioElement(result?.blob ?? null);
    const downloadUrl = audio.src;

    useEffect(() => {
        getModelOptions()
            .then((data) => {
                setVoices(toVoices(data.options.tts ?? []));
                const cur = data.current.tts;
                if (cur) setCurrent(`${cur.engine}|${cur.model}`);
            })
            .catch(() => setError("Could not load voices. Is the API running?"));
    }, []);

    const shown = useMemo(() => {
        const list = showAll ? voices : voices.filter((v) => speaksLanguage(v, language));
        // Installed voices first, then the rest
        return [...list].sort((a, b) => Number(b.installed) - Number(a.installed));
    }, [voices, showAll, language]);

    const selected = voices.find((v) => v.key === current);

    const pick = async (voice: Voice) => {
        if (voice.key === current || !voice.installed) return;
        const previous = current;
        setCurrent(voice.key);
        setSwitching(true);
        setError(null);
        try {
            await switchModel("tts", voice.engine, voice.model);
        } catch (err) {
            setCurrent(previous);
            setError(err instanceof Error ? err.message : "Could not switch voice");
        } finally {
            setSwitching(false);
        }
    };

    const generate = async () => {
        if (!text.trim()) return;
        setGenerating(true);
        setError(null);
        const t0 = performance.now();
        try {
            const blob = await synthesize(text, { language, speed });
            const voiceName = selected?.name ?? "Default voice";
            const seconds = (performance.now() - t0) / 1000;
            const sampleRate = await wavSampleRate(blob).catch(() => null);
            setResult({ blob, voice: voiceName, seconds, sampleRate });
            addClip({ text: text.trim(), voice: voiceName, blob });
        } catch (err) {
            setError(err instanceof Error ? err.message : "Synthesis failed");
        } finally {
            setGenerating(false);
        }
    };

    const earlier = clips.slice(result ? 1 : 0);

    return (
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-start gap-6 px-6 py-8">
            <main className="flex min-w-0 flex-[999_1_560px] flex-col gap-5">
                <div>
                    <h1 className="m-0 text-[26px] font-semibold tracking-tight">Text to speech</h1>
                    <p className="mt-1.5 mb-0 text-[15px] text-muted">
                        Write or paste text, pick a voice, and generate. Long text is split into sentences.
                    </p>
                </div>

                <Panel label="Compose" className="gap-3.5">
                    <label className="flex flex-col gap-2 text-xs text-muted">
                        Text
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            rows={6}
                            maxLength={MAX_CHARS}
                            placeholder={language === "en" ? "Type something to say…" : "Kirjoita jotain sanottavaa…"}
                            className="resize-y rounded-[10px] border border-line-2 bg-bg p-3.5 text-base leading-relaxed text-fg"
                        />
                    </label>
                    <div className="flex flex-wrap items-center gap-4">
                        <label className="flex items-center gap-2.5 text-[13px] text-muted">
                            Speed
                            <input
                                type="range"
                                min={0.5}
                                max={1.5}
                                step={0.05}
                                value={speed}
                                onChange={(e) => setSpeed(Number(e.target.value))}
                                className="w-36 accent-accent"
                            />
                            <span className="font-mono text-fg">{speed.toFixed(2)}×</span>
                        </label>
                        <span className="font-mono text-xs text-faint">
                            {text.length.toLocaleString("fi-FI")} / {MAX_CHARS.toLocaleString("fi-FI")}
                        </span>
                        <button
                            type="button"
                            onClick={generate}
                            disabled={generating || switching || !text.trim()}
                            className={`${buttonPrimary} ml-auto`}
                        >
                            <SpeakerIcon />
                            {generating ? "Generating…" : "Generate speech"}
                        </button>
                    </div>
                    {error && (
                        <p role="alert" className="m-0 text-sm text-bad">
                            {error}
                        </p>
                    )}
                </Panel>

                {result && (
                    <Panel label="Result" className="gap-3.5">
                        <div className="flex flex-wrap items-center gap-3.5">
                            <button
                                type="button"
                                onClick={audio.toggle}
                                aria-label={audio.playing ? "Pause" : "Play"}
                                className="flex h-11 w-11 flex-none items-center justify-center rounded-full bg-accent text-bg hover:bg-accent-hover"
                            >
                                {audio.playing ? <PauseIcon /> : <PlayIcon />}
                            </button>
                            <div className="min-w-0 flex-[1_1_240px]">
                                <Waveform
                                    values={peaks?.values ?? null}
                                    duration={audio.duration || peaks?.duration || 0}
                                    currentTime={audio.currentTime}
                                    onSeek={(t) => audio.seek(t)}
                                    height={44}
                                />
                            </div>
                            <span className="font-mono text-[13px] text-muted">
                                {formatTime(audio.duration || peaks?.duration || 0)}
                            </span>
                            {downloadUrl && (
                                <a href={downloadUrl} download="speechos-tts.wav" className={`${buttonSecondary} no-underline hover:text-fg`}>
                                    Download WAV
                                </a>
                            )}
                            <audio ref={attachAudio} src={audio.src} {...audio.bind} />
                        </div>
                        <p className="m-0 text-xs text-faint">
                            {result.voice} · generated in {result.seconds.toFixed(1)} s
                            {result.sampleRate
                                ? ` · ${(result.sampleRate / 1000).toFixed(result.sampleRate % 1000 ? 2 : 0)} kHz`
                                : ""}
                        </p>
                    </Panel>
                )}

                {earlier.length > 0 && (
                    <section aria-label="History" className="flex flex-col gap-2">
                        <Eyebrow>Earlier in this session</Eyebrow>
                        {earlier.map((c) => (
                            <ClipRow key={c.id} text={c.text} voice={c.voice} blob={c.blob} />
                        ))}
                    </section>
                )}
            </main>

            <aside aria-label="Voices" className="flex min-w-0 flex-[1_1_360px] flex-col gap-3 rounded-xl border border-line bg-surface p-5">
                <div className="flex items-baseline justify-between gap-2">
                    <h2 className="m-0 text-base font-semibold">Voice</h2>
                    <span className="text-xs text-muted">
                        {showAll ? "All languages" : LANGUAGE_NAMES[language]} · {shown.length} voices
                    </span>
                </div>
                <div role="radiogroup" aria-label="Voice" className="flex flex-col gap-2">
                    {shown.map((v) => {
                        const on = v.key === current;
                        return (
                            <button
                                key={v.key}
                                type="button"
                                role="radio"
                                aria-checked={on}
                                onClick={() => pick(v)}
                                disabled={!v.installed || switching}
                                className={`flex w-full flex-col gap-0.5 rounded-[10px] border-[1.5px] p-3.5 text-left text-fg ${
                                    on ? "border-accent bg-selected" : "border-line-2 bg-surface-2 hover:border-line-3"
                                } ${v.installed ? "cursor-pointer" : "cursor-not-allowed opacity-60"}`}
                            >
                                <span className="flex items-center justify-between gap-2">
                                    <span className="text-[15px] font-semibold">{v.name}</span>
                                    <span
                                        aria-hidden="true"
                                        className={`h-4 w-4 flex-none rounded-full border-solid ${on ? "border-[5px] border-accent" : "border-2 border-[#5b6270]"}`}
                                    />
                                </span>
                                <span className="text-xs text-muted">{v.detail}</span>
                                <span className="mt-1 flex flex-wrap gap-1.5">
                                    {(v.installed ? v.tags : ["Not installed", ...v.tags]).map((t) => (
                                        <span key={t} className="rounded-full border border-line-3 px-2 py-0.5 text-[11px] text-[#c9cdd3]">
                                            {t}
                                        </span>
                                    ))}
                                </span>
                                {v.note && <span className="mt-0.5 text-xs text-muted">{v.note}</span>}
                            </button>
                        );
                    })}
                    {voices.length > 0 && shown.length === 0 && (
                        <p className="m-0 text-[13px] text-muted">No voices for this language. Show all languages to pick another.</p>
                    )}
                </div>
                <label className="flex min-h-8 items-center gap-2 text-[13px] text-muted">
                    <input
                        type="checkbox"
                        checked={showAll}
                        onChange={(e) => setShowAll(e.target.checked)}
                        className="h-4 w-4 accent-accent"
                    />
                    Show voices for all languages
                </label>
                <p className="m-0 text-xs leading-relaxed text-faint">
                    Quality notes come from the Finnish benchmark (round-trip character error rate).{" "}
                    <Link href="/models">Install more engines</Link>
                </p>
            </aside>
        </div>
    );
}
