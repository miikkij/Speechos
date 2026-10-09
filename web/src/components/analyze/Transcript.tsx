"use client";

import { useState } from "react";
import { Panel, Skeleton, buttonSecondary } from "@/components/ui";
import { formatTime, toPlainText, type TranscriptLine } from "@/lib/format";

interface TranscriptProps {
    lines: TranscriptLine[] | null;
    loading: boolean;
    currentTime: number;
    speakers: Map<string, { name: string; color: string }>;
    onSeek: (seconds: number) => void;
}

export function Transcript({ lines, loading, currentTime, speakers, onSeek }: TranscriptProps) {
    const [showTimes, setShowTimes] = useState(true);
    const [showSpeakers, setShowSpeakers] = useState(true);
    const [copied, setCopied] = useState(false);

    const hasSpeakers = Boolean(lines?.some((l) => l.speaker));
    const columns = [showTimes ? "44px" : null, showSpeakers && hasSpeakers ? "92px" : null, "minmax(0, 1fr)"]
        .filter(Boolean)
        .join(" ");

    const copy = async () => {
        if (!lines) return;
        await navigator.clipboard.writeText(toPlainText(lines, speakers)).catch(() => undefined);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
    };

    return (
        <Panel label="Transcript" className="gap-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="m-0 text-base font-semibold">Transcript</h2>
                <div className="flex flex-wrap items-center gap-3.5">
                    <label className="flex items-center gap-1.5 text-[13px] text-muted">
                        <input
                            type="checkbox"
                            checked={showTimes}
                            onChange={(e) => setShowTimes(e.target.checked)}
                            className="h-4 w-4 accent-accent"
                        />
                        Timestamps
                    </label>
                    {hasSpeakers && (
                        <label className="flex items-center gap-1.5 text-[13px] text-muted">
                            <input
                                type="checkbox"
                                checked={showSpeakers}
                                onChange={(e) => setShowSpeakers(e.target.checked)}
                                className="h-4 w-4 accent-accent"
                            />
                            Speakers
                        </label>
                    )}
                    <button type="button" onClick={copy} disabled={!lines?.length} className={`${buttonSecondary} min-h-9 text-[13px]`}>
                        {copied ? "Copied" : "Copy text"}
                    </button>
                </div>
            </div>

            {loading && !lines && <Skeleton lines={5} />}
            {!loading && !lines && <p className="m-0 text-[13px] text-muted">Transcript was not requested for this recording.</p>}
            {lines && lines.length === 0 && <p className="m-0 text-[13px] text-muted">No speech was recognized.</p>}

            {lines && lines.length > 0 && (
                <ol className="m-0 flex list-none flex-col gap-0.5 p-0">
                    {lines.map((line, i) => {
                        const active = currentTime >= line.start && currentTime < line.end;
                        const sp = line.speaker ? speakers.get(line.speaker) : undefined;
                        return (
                            <li key={i}>
                                <button
                                    type="button"
                                    onClick={() => onSeek(line.start)}
                                    aria-current={active ? "true" : undefined}
                                    className={`grid w-full cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 text-left text-inherit ${
                                        active ? "border-selected-line bg-selected" : "border-transparent hover:bg-surface-2"
                                    }`}
                                    style={{ gridTemplateColumns: columns }}
                                >
                                    {showTimes && <span className="pt-0.5 font-mono text-xs text-faint">{formatTime(line.start)}</span>}
                                    {showSpeakers && hasSpeakers && (
                                        <span
                                            className="justify-self-start rounded-full px-2 py-0.5 text-xs font-semibold whitespace-nowrap text-bg"
                                            style={{ background: sp?.color ?? "#5b6270" }}
                                        >
                                            {sp?.name ?? "—"}
                                        </span>
                                    )}
                                    <span className="text-[15px] leading-relaxed text-fg">{line.text}</span>
                                </button>
                            </li>
                        );
                    })}
                </ol>
            )}
        </Panel>
    );
}
