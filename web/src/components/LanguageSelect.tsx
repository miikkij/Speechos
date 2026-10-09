"use client";

import { LANGUAGES, type Language } from "@/lib/api";

interface LanguageSelectProps {
    value: Language;
    onChange: (lang: Language) => void;
    disabled?: boolean;
}

export function LanguageSelect({ value, onChange, disabled }: LanguageSelectProps) {
    return (
        <label className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
            Language
            <select
                aria-label="Language"
                value={value}
                onChange={(e) => onChange(e.target.value as Language)}
                disabled={disabled}
                className="text-sm rounded-lg px-2 py-1"
                style={{
                    background: "var(--surface-2)",
                    color: "var(--fg)",
                    border: "1px solid var(--border)",
                }}
            >
                {LANGUAGES.map((l) => (
                    <option key={l.value} value={l.value}>
                        {l.label}
                    </option>
                ))}
            </select>
        </label>
    );
}
