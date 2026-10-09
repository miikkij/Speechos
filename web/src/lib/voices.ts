import type { ModelOption } from "@/lib/api";

export interface Voice {
    key: string;
    engine: string;
    model: string;
    name: string;
    detail: string;
    /** ISO 639-1 codes; "multi" when the engine speaks many languages. */
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

// Notes on Finnish voices come from api/bench (round-trip character error rate, RTF).
const KNOWN: Record<string, Meta> = {
    "piper|fi_FI-harri-medium": {
        name: "Harri",
        languages: ["fi"],
        tags: ["Local", "Fast"],
        note: "Clearest Finnish voice in the benchmark: 3.5% CER, about 30× real-time.",
    },
    "piper|fi_FI-harri-low": {
        name: "Harri (light)",
        languages: ["fi"],
        tags: ["Local", "Low memory"],
        note: "Smaller model for CPU-only machines.",
    },
    "chatterbox|multilingual": {
        name: "Chatterbox Multilingual",
        languages: ["multi"],
        tags: ["Local", "Expressive", "Voice cloning", "GPU"],
        note: "Experimental in Finnish: 4.9–16.5% CER between runs.",
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

function guessLanguages(opt: ModelOption): string[] {
    const m = opt.model.toLowerCase();
    if (m.startsWith("fi_") || m === "fi") return ["fi"];
    if (m.startsWith("en_") || m === "en" || m === "english") return ["en"];
    if (/multilingual|\d+\s*langs?\b|languages/i.test(opt.label)) return ["multi"];
    return ["en"];
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
