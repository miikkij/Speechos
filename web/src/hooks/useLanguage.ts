"use client";

import { useCallback, useSyncExternalStore } from "react";
import { LANGUAGES, type Language } from "@/lib/api";

const STORAGE_KEY = "speechos.language";
const CHANGE_EVENT = "speechos:language";
export const DEFAULT_LANGUAGE: Language = "fi";

function isLanguage(value: string | null): value is Language {
    return LANGUAGES.some((l) => l.value === value);
}

// Fallback when localStorage is unavailable (private mode, blocked site data)
let inMemory: Language | null = null;

function read(): Language {
    try {
        const stored = window.localStorage.getItem(STORAGE_KEY);
        if (isLanguage(stored)) return stored;
    } catch {
        // fall through to the in-memory value
    }
    return inMemory ?? DEFAULT_LANGUAGE;
}

function subscribe(onChange: () => void): () => void {
    window.addEventListener("storage", onChange);
    window.addEventListener(CHANGE_EVENT, onChange);
    return () => {
        window.removeEventListener("storage", onChange);
        window.removeEventListener(CHANGE_EVENT, onChange);
    };
}

/** Selected language (auto / fi / en), remembered in localStorage and shared across tabs. */
export function useLanguage(): [Language, (lang: Language) => void] {
    // Server render and hydration use the default, then the stored value takes over
    const language = useSyncExternalStore(subscribe, read, () => DEFAULT_LANGUAGE);

    const setLanguage = useCallback((lang: Language) => {
        inMemory = lang;
        try {
            window.localStorage.setItem(STORAGE_KEY, lang);
        } catch {
            // ignore — the event below still updates this tab
        }
        window.dispatchEvent(new Event(CHANGE_EVENT));
    }, []);

    return [language, setLanguage];
}
