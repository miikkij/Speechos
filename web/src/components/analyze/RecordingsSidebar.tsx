"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { PlusIcon } from "@/components/Icons";
import { Eyebrow, fieldClass } from "@/components/ui";
import { getRecordings, type Recording } from "@/lib/api";
import { formatDate, formatTime } from "@/lib/format";
import { useApp } from "@/lib/store";

export function RecordingsSidebar({ selectedId }: { selectedId: string | null }) {
    const { recordingsVersion, results } = useApp();
    const [recordings, setRecordings] = useState<Recording[] | null>(null);
    const [query, setQuery] = useState("");

    useEffect(() => {
        getRecordings()
            .then((recs) => setRecordings([...recs].sort((a, b) => b.created_at.localeCompare(a.created_at))))
            .catch(() => setRecordings([]));
    }, [recordingsVersion]);

    const visible = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!recordings || !q) return recordings ?? [];
        return recordings.filter((r) => {
            const name = (r.filename || r.id).toLowerCase();
            const text = results[r.id]?.transcription?.text.toLowerCase() ?? "";
            return name.includes(q) || text.includes(q);
        });
    }, [recordings, query, results]);

    return (
        <aside
            aria-label="Recordings"
            className="flex min-w-0 flex-[1_1_240px] flex-col gap-3 rounded-xl border border-line bg-surface p-4 max-md:order-last"
        >
            <div className="flex items-center justify-between gap-2">
                <Eyebrow>Recordings</Eyebrow>
                <Link
                    href="/capture"
                    className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-[13px] font-semibold text-bg no-underline hover:bg-accent-hover hover:text-bg"
                >
                    <PlusIcon />
                    New
                </Link>
            </div>
            <label className="flex flex-col gap-1 text-xs text-muted">
                Search
                <input
                    type="search"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Name or words in transcript"
                    className={`${fieldClass} px-2.5 py-2 text-[13px]`}
                />
            </label>
            {recordings === null && <p className="m-0 text-[13px] text-muted">Loading…</p>}
            {recordings?.length === 0 && (
                <p className="m-0 text-[13px] text-muted">
                    No recordings yet. <Link href="/capture">Record or upload one</Link>.
                </p>
            )}
            <ul className="m-0 flex max-h-72 list-none flex-col gap-1 overflow-y-auto p-0 md:max-h-[calc(100vh-240px)]">
                {visible.map((rec) => {
                    const on = rec.id === selectedId;
                    return (
                        <li key={rec.id}>
                            <Link
                                href={`/analyze?r=${encodeURIComponent(rec.id)}`}
                                aria-current={on ? "true" : undefined}
                                className={`flex min-w-0 flex-col gap-0.5 rounded-lg border px-3 py-2.5 text-fg no-underline hover:text-fg ${
                                    on ? "border-selected-line bg-nav" : "border-transparent hover:bg-surface-2"
                                }`}
                            >
                                <span className="truncate font-medium">{rec.filename || rec.id}</span>
                                <span className="flex gap-2 font-mono text-xs text-muted">
                                    <span>{formatDate(rec.created_at)}</span>
                                    <span>{formatTime(rec.duration)}</span>
                                    {results[rec.id]?.transcription?.language && (
                                        <span>{results[rec.id]?.transcription?.language.toUpperCase()}</span>
                                    )}
                                </span>
                            </Link>
                        </li>
                    );
                })}
            </ul>
        </aside>
    );
}
