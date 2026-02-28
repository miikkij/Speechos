"use client";

import { useEffect, useState } from "react";
import { synthesize, getModelOptions, switchModel, type ModelOption } from "@/lib/api";
import { loadSettings, saveSettings } from "@/lib/storage";
import { AudioPlayer } from "./AudioPlayer";

interface EngineGuide {
    title: string;
    description: string;
    examples: { label: string; text: string }[];
}

const ENGINE_GUIDES: Record<string, EngineGuide> = {
    "qwen3-tts": {
        title: "Qwen3-TTS: Natural language emotions",
        description: `Describe emotions naturally in your text. The model interprets tone from context.
Voices: alloy, echo, fable, nova, onyx, shimmer
No special markup needed, just write expressively.`,
        examples: [
            { label: "Enthusiastic", text: "I'm so excited about this! This is the best day of my life, I can't wait to tell everyone!" },
            { label: "Sad", text: "I can't believe this happened... I really thought things would be different this time." },
            { label: "Laughing", text: "Ha ha, that's hilarious! Oh man, I haven't laughed this hard in ages!" },
            { label: "Pleading", text: "Please, I need your help. I don't know what to do anymore and you're the only one who can help me." },
            { label: "Angry", text: "This is absolutely unacceptable! I've been waiting for three hours and nobody has even bothered to explain what's going on!" },
            { label: "Whispering", text: "Hey... come closer... I have a secret to tell you, but you have to promise not to tell anyone." },
        ],
    },
    chattts: {
        title: "ChatTTS: Token tags",
        description: `Insert special tags to control speech:
  [laugh_0]-[laugh_2]  laughter (0=subtle, 2=strong)
  [break_0]-[break_7]  pause (0=short, 7=long)
  [oral_0]-[oral_9]    oral filler (um, uh)
  [uv_break]           micro-pause`,
        examples: [
            { label: "Casual with filler", text: "Well [oral_2] I think [break_1] that's a really good point actually." },
            { label: "Laughing", text: "And then he just [break_1] fell right off the chair [laugh_2] I couldn't stop laughing!" },
            { label: "Thoughtful pause", text: "You know [oral_1] I've been thinking about this [break_3] and I think [break_1] we should go for it." },
            { label: "Hesitant", text: "[oral_3] I'm not [uv_break] entirely sure about this [break_2] but [oral_1] maybe we could try?" },
        ],
    },
    orpheus: {
        title: "Orpheus: XML emotion tags",
        description: `Wrap text with XML tags for emotional expressions:
  <laugh>, <chuckle>, <sigh>, <cough>,
  <gasp>, <yawn>, <groan>`,
        examples: [
            { label: "Laughing story", text: "So then he said <laugh>ha ha ha</laugh> I couldn't believe it <gasp>really?!</gasp>" },
            { label: "Tired morning", text: "<yawn>Good morning everyone</yawn> <sigh>I barely slept last night</sigh> but let's get started." },
            { label: "Surprised", text: "Wait, you're telling me <gasp>she actually did it?!</gasp> <laugh>No way!</laugh> That's incredible!" },
            { label: "Reluctant", text: "<sigh>Fine, I'll do it</sigh> but <groan>this is going to be a long day</groan>" },
        ],
    },
    parler: {
        title: "Parler: Voice descriptions",
        description: `Set a voice description to control speaker style.
The model generates speech matching your description.`,
        examples: [
            { label: "Excited male", text: "Welcome to the show, folks! Tonight we have an incredible lineup that you absolutely do not want to miss!" },
            { label: "Calm female", text: "Take a deep breath and relax. Let your shoulders drop, and feel the tension melting away." },
            { label: "Formal narrator", text: "In the year eighteen sixty-five, a remarkable discovery was made in the remote highlands of Scotland." },
            { label: "Cheerful", text: "Good morning sunshine! What a beautiful day to be alive! Let's make the most of every single moment!" },
        ],
    },
    piper: {
        title: "Piper: Voice list",
        description: `Voices: lessac (F), amy (F), ryan (M), arctic (M), alan (British M), libritts
No emotion markup. Fast CPU inference.`,
        examples: [
            { label: "News anchor", text: "Good evening. Tonight's top story: local researchers have made a groundbreaking discovery in renewable energy technology." },
            { label: "Audiobook", text: "The old house stood at the end of the lane, its windows dark, its garden overgrown with wild roses and tangled ivy." },
            { label: "Instructions", text: "First, preheat your oven to three hundred and fifty degrees. Then, combine the flour, sugar, and butter in a large mixing bowl." },
        ],
    },
    kokoro: {
        title: "Kokoro: Voice IDs",
        description: `Voice IDs: af_heart (F), am_adam (M), bf_emma (British F), bm_george (British M)
No emotion markup. High-quality neural TTS.`,
        examples: [
            { label: "Storytelling", text: "Once upon a time, in a land far away, there lived a young adventurer who dreamed of sailing across the great ocean." },
            { label: "Presentation", text: "Thank you all for coming today. I'd like to share some exciting developments from our latest research." },
            { label: "Conversational", text: "So I was thinking, maybe this weekend we could try that new restaurant downtown? I heard they have amazing pasta." },
        ],
    },
    bark: {
        title: "Bark: Special characters",
        description: `\u266A ... \u266A for singing, ALL CAPS for emphasis, ... for pauses.
Note: Bark is slow. GPU strongly recommended.`,
        examples: [
            { label: "Singing", text: "\u266A Twinkle twinkle little star, how I wonder what you are \u266A" },
            { label: "Emphatic", text: "I am ABSOLUTELY certain that this is the RIGHT thing to do... trust me on this one." },
            { label: "Dramatic", text: "And then... silence... nothing but the sound of the wind... and then BOOM! Everything changed." },
        ],
    },
    espeak: {
        title: "eSpeak: Robotic voice",
        description: `Robotic/formant synthesis. Instant response.
Supports 100+ languages. No emotion control.`,
        examples: [
            { label: "System message", text: "Attention: system diagnostics complete. All modules operating within normal parameters." },
            { label: "Countdown", text: "Initiating launch sequence. Ten. Nine. Eight. Seven. Six. Five. Four. Three. Two. One. Liftoff." },
        ],
    },
};

function getEngineFromVoice(voice: string): string {
    return voice.split("|")[0] ?? "";
}

function SynthWaveform() {
    return (
        <div className="flex items-end gap-[3px] h-5">
            {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                <div
                    key={i}
                    className={`synth-bar synth-bar-${i}`}
                    style={{ background: "var(--accent)" }}
                />
            ))}
        </div>
    );
}

export function TtsPlayground() {
    const [text, setText] = useState("");
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [loading, setLoading] = useState(false);
    const [switching, setSwitching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [ttsOptions, setTtsOptions] = useState<ModelOption[]>([]);
    const [selectedVoice, setSelectedVoice] = useState("");
    const [showGuide, setShowGuide] = useState(false);

    useEffect(() => {
        getModelOptions()
            .then(async (data) => {
                const opts = data.options.tts || [];
                setTtsOptions(opts);
                const cur = data.current.tts;
                const serverVoice = cur ? `${cur.engine}|${cur.model}` : "";

                // Restore saved voice preference
                const savedVoice = loadSettings().ttsVoice;
                if (savedVoice && opts.some((o) => `${o.engine}|${o.model}` === savedVoice)) {
                    setSelectedVoice(savedVoice);
                    if (savedVoice !== serverVoice) {
                        const [engine, model] = savedVoice.split("|");
                        try {
                            await switchModel("tts", engine, model);
                        } catch {
                            setSelectedVoice(serverVoice);
                        }
                    }
                } else if (cur) {
                    setSelectedVoice(serverVoice);
                }
            })
            .catch(() => { });
    }, []);

    const handleVoiceChange = async (value: string) => {
        setSelectedVoice(value);
        saveSettings({ ttsVoice: value });
        const [engine, model] = value.split("|");
        setSwitching(true);
        setError(null);
        try {
            await switchModel("tts", engine, model);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to switch voice");
        } finally {
            setSwitching(false);
        }
    };

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

    const currentEngine = getEngineFromVoice(selectedVoice);
    const guide = ENGINE_GUIDES[currentEngine];

    return (
        <div className="space-y-4">
            <h2
                className="text-lg font-bold tracking-wide"
                style={{ fontFamily: "var(--font-display), serif" }}
            >
                Text to Speech
            </h2>

            {/* Voice selector */}
            {ttsOptions.length > 0 && (
                <div className="flex items-center gap-2">
                    <label className="text-xs shrink-0" style={{ color: "var(--muted)" }}>
                        Voice
                    </label>
                    <select
                        value={selectedVoice}
                        onChange={(e) => handleVoiceChange(e.target.value)}
                        disabled={switching || loading}
                        className="flex-1 text-sm rounded-lg px-3 py-1.5 truncate"
                        style={{
                            background: "var(--surface-2)",
                            color: "var(--fg)",
                            border: "1px solid var(--border)",
                        }}
                    >
                        {ttsOptions.map((opt) => (
                            <option key={`${opt.engine}|${opt.model}`} value={`${opt.engine}|${opt.model}`}>
                                {opt.label}
                            </option>
                        ))}
                    </select>
                    {switching && (
                        <span className="text-xs animate-pulse" style={{ color: "var(--warning)" }}>Switching...</span>
                    )}
                </div>
            )}

            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text to synthesize..."
                rows={3}
                maxLength={10000}
                className="w-full rounded-lg p-3 text-sm resize-y outline-none transition-all"
                style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    color: "var(--fg)",
                }}
            />

            {/* Markup guide toggle */}
            <button
                onClick={() => setShowGuide(!showGuide)}
                className="text-xs px-3 py-1 rounded-lg transition-colors"
                style={{
                    color: "var(--muted)",
                    background: showGuide ? "var(--surface-2)" : "transparent",
                    border: "1px solid var(--border)",
                }}
            >
                {showGuide ? "Hide markup guide" : "Markup guide"}
            </button>

            {showGuide && (
                <div
                    className="rounded-lg p-4 space-y-3"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                >
                    <h4 className="text-xs font-medium" style={{ color: "var(--accent)" }}>
                        {guide?.title ?? `${currentEngine || "Unknown"} engine`}
                    </h4>
                    <pre
                        className="text-xs whitespace-pre-wrap leading-relaxed"
                        style={{ color: "var(--muted)" }}
                    >
                        {guide?.description ?? "No special markup documented for this engine."}
                    </pre>

                    {/* Clickable examples */}
                    {guide && guide.examples.length > 0 && (
                        <div className="space-y-2 pt-1" style={{ borderTop: "1px solid var(--border)" }}>
                            <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>
                                Examples
                            </span>
                            {guide.examples.map((ex) => (
                                <div
                                    key={ex.label}
                                    className="group flex items-start gap-2 rounded-md p-2 transition-colors cursor-pointer"
                                    style={{ background: "var(--surface-2)" }}
                                    onClick={() => setText(ex.text)}
                                >
                                    <div className="flex-1 min-w-0">
                                        <span
                                            className="text-xs font-medium block mb-0.5"
                                            style={{ color: "var(--accent)" }}
                                        >
                                            {ex.label}
                                        </span>
                                        <span
                                            className="text-xs block leading-relaxed"
                                            style={{ color: "var(--muted)" }}
                                        >
                                            {ex.text}
                                        </span>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setText(ex.text);
                                        }}
                                        className="shrink-0 text-xs px-2 py-1 rounded transition-all opacity-60 hover:opacity-100"
                                        style={{
                                            color: "var(--accent)",
                                            border: "1px solid var(--border)",
                                            background: "var(--surface)",
                                        }}
                                    >
                                        Try it
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            <div className="flex items-center gap-3">
                <button
                    onClick={handleSynthesize}
                    disabled={loading || !text.trim() || switching}
                    className="px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50"
                    style={{
                        background: "var(--accent)",
                        color: "#fff",
                        boxShadow: loading ? "0 0 20px rgba(232, 145, 58, 0.3)" : "none",
                    }}
                >
                    {loading ? "Synthesizing..." : "Speak"}
                </button>

                {/* Synthesis waveform animation */}
                {loading && <SynthWaveform />}

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
