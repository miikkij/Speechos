const STORAGE_KEY = "speechos-settings";

export interface PersistedSettings {
    micDeviceId?: string;
    audioSettings?: {
        sampleRate: number;
        gain: number;
        noiseSuppression: boolean;
        echoCancellation: boolean;
        autoGainControl: boolean;
    };
    models?: Record<string, { engine: string; model: string }>;
    ttsVoice?: string;
    playbackVolume?: number;
}

export function loadSettings(): PersistedSettings {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        return JSON.parse(raw) as PersistedSettings;
    } catch {
        return {};
    }
}

export function saveSettings(partial: Partial<PersistedSettings>): void {
    try {
        const existing = loadSettings();
        const merged = { ...existing, ...partial };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
    } catch {
        // localStorage unavailable (SSR, private browsing quota exceeded)
    }
}
