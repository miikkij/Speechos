"use client";

import { useEffect, useState } from "react";
import type { SystemInfo } from "@/lib/api";
import { getSystemInfo, checkHealth } from "@/lib/api";

export function StatusBar() {
    const [online, setOnline] = useState<boolean | null>(null);
    const [info, setInfo] = useState<SystemInfo | null>(null);

    useEffect(() => {
        let mounted = true;

        const poll = async () => {
            const healthy = await checkHealth();
            if (!mounted) return;
            setOnline(healthy);
            if (healthy) {
                try {
                    const data = await getSystemInfo();
                    if (mounted) setInfo(data);
                } catch { /* ignore */ }
            }
        };

        poll();
        const id = setInterval(poll, 10_000);
        return () => {
            mounted = false;
            clearInterval(id);
        };
    }, []);

    return (
        <div
            className="flex items-center gap-4 px-4 py-2 text-xs"
            style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)" }}
        >
            <div className="flex items-center gap-1.5">
                <div
                    className="w-2 h-2 rounded-full"
                    style={{
                        background:
                            online === null ? "var(--warning)" : online ? "var(--success)" : "var(--error)",
                    }}
                />
                <span style={{ color: "var(--muted)" }}>
                    {online === null ? "Checking..." : online ? "API Connected" : "API Offline"}
                </span>
            </div>

            {info && (
                <>
                    <span style={{ color: "var(--border)" }}>|</span>
                    <span style={{ color: "var(--muted)" }}>
                        {info.hardware.tier}
                    </span>
                    {info.hardware.gpu_name && (
                        <>
                            <span style={{ color: "var(--border)" }}>|</span>
                            <span style={{ color: "var(--muted)" }}>
                                {info.hardware.gpu_name} ({info.hardware.vram_gb}GB)
                            </span>
                        </>
                    )}
                    {info.models.loaded.length > 0 && (
                        <>
                            <span style={{ color: "var(--border)" }}>|</span>
                            <span style={{ color: "var(--muted)" }}>
                                Models: {info.models.loaded.join(", ")}
                            </span>
                        </>
                    )}
                </>
            )}
        </div>
    );
}
