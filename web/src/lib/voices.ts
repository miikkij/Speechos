import type { ModelOption } from "@/lib/api";

export interface Voice {
    key: string;
    engine: string;
    model: string;
    name: string;
    detail: string;
    /** ISO 639-1 codes; "multi" = verified to speak Finnish and English; "other" = not verified. */
    languages: string[];
    tags: string[];
    note: string;
    installed: boolean;
}

interface Meta {
    name: string;
    languages: string[];
    tags: string[];
    note: string;
}

// Notes on Finnish voices come from api/bench (2026-10: round-trip CER over 3 seeds, RTF on an RTX 4090).
const KNOWN: Record<string, Meta> = {
    "piper|fi_FI-harri-medium": {
        name: "Harri",
        languages: ["fi"],
        tags: ["Local", "Fast"],
        note: "Fast and clear: 4.7% CER in the Finnish benchmark, about 30× real-time. Can stumble on very short replies.",
    },
    "piper|fi_FI-harri-low": {
        name: "Harri (light)",
        languages: ["fi"],
        tags: ["Local", "Low memory"],
        note: "Smaller model for CPU-only machines.",
    },
    "chatterbox-fi|sample": {
        name: "Chatterbox-Finnish",
        languages: ["fi"],
        tags: ["Local", "Natural", "Voice cloning", "GPU"],
        note: "Most natural local Finnish voice: 3.3% CER, 0.43× real-time on an RTX 4090. About 4.4 GB of VRAM.",
    },
    "chatterbox|multilingual": {
        name: "Chatterbox Multilingual",
        languages: ["multi"],
        tags: ["Local", "Expressive", "Voice cloning", "GPU"],
        note: "Expressive, but needs a Finnish reference voice: 4.0% CER with one, 17% without. Slower than real-time.",
    },
    "espeak|fi": {
        name: "eSpeak Finnish",
        languages: ["fi"],
        tags: ["Local", "Tiny"],
        note: "Robotic, but always available.",
    },
    "espeak|en": { name: "eSpeak English", languages: ["en"], tags: ["Local", "Tiny"], note: "Robotic, but always available." },
    "kokoro|default": { name: "Kokoro", languages: ["en"], tags: ["Local", "Natural"], note: "High quality for its size. English only." },
    "piper|en_US-lessac-medium": { name: "Lessac", languages: ["en"], tags: ["Local", "Fast"], note: "Default English voice." },
};

/**
 * Only Finnish/English codes in the model name are trusted. A label like "17 langs" says nothing about
 * Finnish (XTTS-v2 has no Finnish), so other engines are listed under "all languages" only.
 */
function guessLanguages(opt: ModelOption): string[] {
    const m = opt.model.toLowerCase();
    if (m.startsWith("fi_") || m === "fi") return ["fi"];
    if (m.startsWith("en_") || m === "en" || m === "english") return ["en"];
    return ["other"];
}

/** Turns the API's TTS options into voice cards. */
export function toVoices(options: ModelOption[]): Voice[] {
    return options.map((opt) => {
        const key = `${opt.engine}|${opt.model}`;
        const meta = KNOWN[key];
        const docker = opt.label.startsWith("[Docker]");
        const label = opt.label.replace("[Docker] ", "");
        const [title, paren] = label.split(" (");
        return {
            key,
            engine: opt.engine,
            model: opt.model,
            name: meta?.name ?? title,
            detail: `${opt.engine} · ${opt.model}`,
            languages: meta?.languages ?? guessLanguages(opt),
            tags: meta?.tags ?? [docker ? "Docker" : "Local"],
            note: meta?.note ?? (paren ? paren.replace(/\)$/, "") : ""),
            installed: opt.installed !== false,
        };
    });
}

export function speaksLanguage(voice: Voice, language: string): boolean {
    if (language === "auto") return true;
    return voice.languages.includes(language) || voice.languages.includes("multi");
}
