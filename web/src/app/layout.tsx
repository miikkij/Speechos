import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { AppShell } from "@/components/AppShell";
import { AppProvider } from "@/lib/store";
import "./globals.css";

// next/font downloads these at build time and serves them locally — no runtime requests to Google.
const plexSans = IBM_Plex_Sans({
    subsets: ["latin"],
    weight: ["400", "500", "600"],
    variable: "--font-plex-sans",
});

const plexMono = IBM_Plex_Mono({
    subsets: ["latin"],
    weight: ["400", "500"],
    variable: "--font-plex-mono",
});

export const metadata: Metadata = {
    title: "Speechos",
    description: "Local-first speech analysis and synthesis platform",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
            <body>
                <AppProvider>
                    <AppShell>{children}</AppShell>
                </AppProvider>
            </body>
        </html>
    );
}
