"use client";

import { useCallback, useEffect, useState } from "react";
import { loadSettings, saveSettings } from "@/lib/storage";

export interface AudioDevice {
    deviceId: string;
    label: string;
}

export interface UseAudioDevicesReturn {
    devices: AudioDevice[];
    selectedId: string;
    setSelectedId: (id: string) => void;
    refresh: () => Promise<void>;
}

export function useAudioDevices(): UseAudioDevicesReturn {
    const [devices, setDevices] = useState<AudioDevice[]>([]);
    const [selectedId, setSelectedIdRaw] = useState("");

    const setSelectedId = useCallback((id: string) => {
        setSelectedIdRaw(id);
        saveSettings({ micDeviceId: id });
    }, []);

    const refresh = useCallback(async () => {
        try {
            // Request permission first; enumerateDevices only returns labels
            // after the user has granted mic access at least once
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach((t) => t.stop());

            const all = await navigator.mediaDevices.enumerateDevices();
            const mics = all
                .filter((d) => d.kind === "audioinput")
                .map((d, i) => ({
                    deviceId: d.deviceId,
                    label: d.label || `Microphone ${i + 1}`,
                }));
            setDevices(mics);

            const saved = loadSettings().micDeviceId;
            setSelectedIdRaw((prev) => {
                // Prefer saved device, then current selection, then first device
                if (saved && mics.some((m) => m.deviceId === saved)) return saved;
                if (prev && mics.some((m) => m.deviceId === prev)) return prev;
                return mics[0]?.deviceId ?? "";
            });
        } catch {
            // Permission denied or no devices
            setDevices([]);
        }
    }, []);

    useEffect(() => {
        refresh();

        // Listen for device changes (plug/unplug)
        const handler = () => { refresh(); };
        navigator.mediaDevices.addEventListener("devicechange", handler);
        return () => navigator.mediaDevices.removeEventListener("devicechange", handler);
    }, [refresh]);

    return { devices, selectedId, setSelectedId, refresh };
}
