"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { LogoIcon } from "@/components/Icons";
import { LanguageSelect } from "@/components/LanguageSelect";
import { checkHealth, getSystemInfo, type SystemInfo } from "@/lib/api";
import { useApp } from "@/lib/store";

const NAV = [
    { href: "/capture", label: "Capture" },
    { href: "/analyze", label: "Analyze" },
    { href: "/synthesize", label: "Synthesize" },
    { href: "/models", label: "Models" },
];

function StatusPill() {
    const [online, setOnline] = useState<boolean | null>(null);
    const [info, setInfo] = useState<SystemInfo | null>(null);

    useEffect(() => {
        let alive = true;
        const poll = async () => {
            const ok = await checkHealth();
            if (!alive) return;
            setOnline(ok);
            if (ok) {
                getSystemInfo()
                    .then((i) => alive && setInfo(i))
                    .catch(() => undefined);
            }
        };
        poll();
        const id = setInterval(poll, 10_000);
        return () => {
            alive = false;
            clearInterval(id);
        };
    }, []);

    const hw = info?.hardware;
    let text = "Connecting…";
    if (online === false) text = "Offline — start the API";
    else if (online && hw) {
        const device = hw.gpu_available ? `GPU ${hw.vram_gb} GB` : `CPU ${hw.ram_gb} GB RAM`;
        text = `Online · ${device} · ${hw.tier}`;
    } else if (online) text = "Online";

    const dot = online === false ? "bg-bad" : online ? "bg-ok" : "bg-faint";
    return (
        <div
            role="status"
            className="flex items-center gap-2 rounded-full border border-line-2 px-3 py-1.5 text-xs text-muted"
        >
            <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
            <span>{text}</span>
        </div>
    );
}

export function AppShell({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const { language, setLanguage } = useApp();

    return (
        <div className="min-h-screen bg-bg text-fg">
            <header className="flex flex-wrap items-center gap-x-7 gap-y-4 border-b border-line bg-chrome px-6 py-3.5">
                <Link href="/capture" className="flex items-center gap-2.5 text-fg no-underline hover:text-fg">
                    <LogoIcon className="text-accent" />
                    <span className="text-base font-semibold tracking-tight">Speechos</span>
                </Link>
                <nav aria-label="Main" className="flex flex-wrap gap-1">
                    {NAV.map((item) => {
                        const active = pathname.startsWith(item.href);
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                aria-current={active ? "page" : undefined}
                                className={`rounded-lg px-3.5 py-2 font-medium no-underline ${
                                    active ? "bg-nav text-fg hover:text-fg" : "text-muted hover:text-fg"
                                }`}
                            >
                                {item.label}
                            </Link>
                        );
                    })}
                </nav>
                <div className="ml-auto flex flex-wrap items-center gap-3">
                    <LanguageSelect value={language} onChange={setLanguage} />
                    <StatusPill />
                </div>
            </header>
            {children}
        </div>
    );
}
