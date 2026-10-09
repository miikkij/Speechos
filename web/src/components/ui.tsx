/** Small shared building blocks for the studio theme. */

export const buttonSecondary =
    "inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-line-2 bg-surface-2 px-3.5 font-medium text-fg hover:border-line-3 disabled:opacity-50";

export const buttonPrimary =
    "inline-flex min-h-11 items-center gap-2 rounded-[10px] bg-accent px-5 text-[15px] font-semibold text-bg hover:bg-accent-hover disabled:opacity-50";

export const fieldClass = "rounded-lg border border-line-2 bg-surface-2 text-fg";

export function Panel({
    children,
    className = "",
    label,
}: {
    children: React.ReactNode;
    className?: string;
    label?: string;
}) {
    return (
        <section aria-label={label} className={`flex flex-col rounded-xl border border-line bg-surface p-5 ${className}`}>
            {children}
        </section>
    );
}

export function Eyebrow({ children, as: Tag = "h2" }: { children: React.ReactNode; as?: "h2" | "h3" | "span" }) {
    return <Tag className="m-0 text-[13px] font-semibold uppercase tracking-[0.04em] text-muted">{children}</Tag>;
}

export function MeterRow({
    label,
    value,
    fraction,
    color,
    strong,
    labelWidth = "84px",
    valueWidth = "44px",
}: {
    label: React.ReactNode;
    value: string;
    fraction: number;
    color: string;
    strong?: boolean;
    labelWidth?: string;
    valueWidth?: string;
}) {
    return (
        <div
            className="grid items-center gap-2.5 text-[13px]"
            style={{ gridTemplateColumns: `${labelWidth} minmax(0, 1fr) ${valueWidth}` }}
        >
            <span className={strong ? "font-semibold text-fg" : "text-muted"}>{label}</span>
            <span className="h-2 overflow-hidden rounded bg-line">
                <span
                    className="block h-full rounded"
                    style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%`, background: color }}
                />
            </span>
            <span className="text-right font-mono text-xs text-muted">{value}</span>
        </div>
    );
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
    return (
        <div className="flex animate-pulse flex-col gap-3" aria-hidden="true">
            {Array.from({ length: lines }, (_, i) => (
                <div key={i} className="h-4 rounded bg-line" style={{ width: `${90 - i * 18}%` }} />
            ))}
        </div>
    );
}
