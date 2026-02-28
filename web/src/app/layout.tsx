import type { Metadata } from "next";
import { Playfair_Display } from "next/font/google";
import "./globals.css";

const playfair = Playfair_Display({
    subsets: ["latin"],
    variable: "--font-display",
    weight: ["400", "700"],
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
        <html lang="en" className={playfair.variable}>
            <body>{children}</body>
        </html>
    );
}
