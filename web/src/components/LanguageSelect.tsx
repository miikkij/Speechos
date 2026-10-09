"use client";

import { LANGUAGES, type Language } from "@/lib/api";

interface LanguageSelectProps {
    value: Language;
    onChange: (lang: Language) => void;
    disabled?: boolean;
    label?: string;
    /** "inline" sits in the header; "stacked" puts the label above a full-width field. */
    layout?: "inline" | "stacked";
}

export function LanguageSelect({ value, onChange, disabled, label = "Language", layout = "inline" }: LanguageSelectProps) {
    const stacked = layout === "stacked";
    return (
        <label className={`flex text-muted ${stacked ? "flex-col gap-1.5 text-xs" : "items-center gap-2 text-[13px]"}`}>
            {label}
            <select
                value={value}
                onChange={(e) => onChange(e.target.value as Language)}
                disabled={disabled}
                className={`rounded-lg border border-line-2 bg-surface-2 text-fg ${stacked ? "p-2.5 text-sm" : "px-2.5 py-1.5"}`}
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
