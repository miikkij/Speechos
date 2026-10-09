"use client";

import { useCallback, useEffect, useState } from "react";
import { LANGUAGES, type Language } from "@/lib/api";

const STORAGE_KEY = "speechos.language";
export const DEFAULT_LANGUAGE: Language = "fi";

function isLanguage(value: string | null): value is Language {
    return LANGUAGES.some((l) => l.value === value);
}

/** Selected UI language (auto / fi / en), remembered in localStorage. */
export function useLanguage(): [Language, (lang: Language) => void] {
    const [language, setLanguageState] = useState<Language>(DEFAULT_LANGUAGE);

    // Read after mount so server and first client render agree
    useEffect(() => {
        try {
            const stored = window.localStorage.getItem(STORAGE_KEY);
            if (isLanguage(stored)) setLanguageState(stored);
        } catch {
            // Storage unavailable (private mode, blocked) — keep the default
        }
    }, []);

    const setLanguage = useCallback((lang: Language) => {
        setLanguageState(lang);
        try {
            window.localStorage.setItem(STORAGE_KEY, lang);
        } catch {
            // ignore
        }
    }, []);

    return [language, setLanguage];
}
