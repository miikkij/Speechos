"use client";

import { useCallback, useEffect, useState } from "react";
import {
    getRecordings,
    getRecordingBlob,
    deleteRecording,
    type Recording,
} from "@/lib/api";

interface RecordingsLibraryProps {
    onSelect: (blob: Blob, recording: Recording) => void;
    refreshKey?: number;
}

export function RecordingsLibrary({ onSelect, refreshKey }: RecordingsLibraryProps) {
    const [recordings, setRecordings] = useState<Recording[]>([]);
    const [loading, setLoading] = useState(false);
    const [loadingId, setLoadingId] = useState<string | null>(null);
    const [expanded, setExpanded] = useState(false);

    const refresh = useCallback(() => {
        setLoading(true);
        getRecordings()
            .then(setRecordings)
            .catch(() => { })
            .finally(() => setLoading(false));
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh, refreshKey]);

    const handleSelect = async (rec: Recording) => {
        setLoadingId(rec.id);
        try {
            const blob = await getRecordingBlob(rec.id);
            onSelect(blob, rec);
        } catch {
            // ignore
        } finally {
            setLoadingId(null);
        }
    };

    const handleDelete = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        try {
            await deleteRecording(id);
            setRecordings((prev) => prev.filter((r) => r.id !== id));
        } catch {
            // ignore
        }
    };

    if (recordings.length === 0 && !loading) return null;

    return (
        <div
            className="rounded-lg overflow-hidden"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium"
                style={{ color: "var(--fg)" }}
            >
                <span>
                    Saved Recordings
                    <span
                        className="ml-2 text-xs px-2 py-0.5 rounded-full"
                        style={{ background: "var(--surface-2)", color: "var(--muted)" }}
                    >
                        {recordings.length}
                    </span>
                </span>
                <span style={{ color: "var(--muted)" }}>{expanded ? "▲" : "▼"}</span>
            </button>

            {expanded && (
                <div className="border-t" style={{ borderColor: "var(--border)" }}>
                    {loading ? (
                        <div className="p-4 text-xs text-center" style={{ color: "var(--muted)" }}>
                            Loading...
                        </div>
                    ) : (
                        <div className="max-h-64 overflow-y-auto">
                            {recordings.map((rec) => (
                                <div
                                    key={rec.id}
                                    onClick={() => handleSelect(rec)}
                                    className="flex items-center gap-3 px-4 py-2.5 cursor-pointer transition-colors hover:opacity-80"
                                    style={{
                                        borderBottom: "1px solid var(--border)",
                                        opacity: loadingId === rec.id ? 0.6 : 1,
                                    }}
                                >
                                    <span className="text-sm" style={{ color: "var(--accent)" }}>
                                        ▶
                                    </span>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-xs font-mono truncate" style={{ color: "var(--fg)" }}>
                                            {rec.id}
                                        </div>
                                        <div className="text-xs" style={{ color: "var(--muted)" }}>
                                            {rec.duration.toFixed(1)}s &middot;{" "}
                                            {(rec.size_bytes / 1024).toFixed(0)} KB &middot;{" "}
                                            {new Date(rec.created_at).toLocaleDateString()}
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => handleDelete(e, rec.id)}
                                        className="text-xs px-2 py-1 rounded transition-colors"
                                        style={{ color: "var(--error)" }}
                                        title="Delete recording"
                                    >
                                        ✕
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
