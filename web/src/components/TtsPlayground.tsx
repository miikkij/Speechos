"use client";

import { useState } from "react";
import { synthesize } from "@/lib/api";
import { AudioPlayer } from "./AudioPlayer";

export function TtsPlayground() {
  const [text, setText] = useState("");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSynthesize = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const blob = await synthesize(text);
      setAudioBlob(blob);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Synthesis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">Text to Speech</h2>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to synthesize..."
        rows={3}
        maxLength={10000}
        className="w-full rounded-lg p-3 text-sm resize-y outline-none focus:ring-1"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          color: "var(--fg)",
        }}
      />

      <div className="flex items-center gap-3">
        <button
          onClick={handleSynthesize}
          disabled={loading || !text.trim()}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          style={{ background: "var(--accent)", color: "#fff" }}
        >
          {loading ? "Synthesizing..." : "Speak"}
        </button>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {text.length}/10000
        </span>
      </div>

      {error && (
        <p className="text-sm" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}

      {audioBlob && <AudioPlayer src={audioBlob} label="Synthesized speech" />}
    </div>
  );
}
