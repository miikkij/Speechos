interface IconProps {
    size?: number;
    className?: string;
}

const stroke = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
};

export function LogoIcon({ size = 22, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke}>
            <path d="M3 12h2M7 8v8M11 4v16M15 7v10M19 10v4" />
        </svg>
    );
}

export function PlayIcon({ size = 18, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
            <path d="M7 4.5v15l13-7.5z" />
        </svg>
    );
}

export function PauseIcon({ size = 18, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
            <rect x="6" y="4.5" width="4" height="15" rx="1" />
            <rect x="14" y="4.5" width="4" height="15" rx="1" />
        </svg>
    );
}

export function StopIcon({ size = 30, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
    );
}

export function MicIcon({ size = 34, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke}>
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
            <path d="M12 19v3" />
        </svg>
    );
}

export function PlusIcon({ size = 14, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke} strokeWidth={2.4}>
            <path d="M12 5v14M5 12h14" />
        </svg>
    );
}

export function RefreshIcon({ size = 16, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke}>
            <path d="M21 12a9 9 0 1 1-3-6.7L21 8" />
            <path d="M21 3v5h-5" />
        </svg>
    );
}

export function DownloadIcon({ size = 16, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke}>
            <path d="M12 3v12M7 10l5 5 5-5M5 21h14" />
        </svg>
    );
}

export function UploadIcon({ size = 28, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke} strokeWidth={1.8}>
            <path d="M12 16V4M7 9l5-5 5 5" />
            <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
        </svg>
    );
}

export function SpeakerIcon({ size = 18, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke} strokeWidth={2.2}>
            <path d="M11 5 6 9H3v6h3l5 4z" />
            <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14" />
        </svg>
    );
}

export function TrashIcon({ size = 16, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" {...stroke}>
            <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />
        </svg>
    );
}
