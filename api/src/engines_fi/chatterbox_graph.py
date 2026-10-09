"""CUDA-graph decoder for Chatterbox T3 (bundled chatterbox_ code) — same sampling, no per-kernel launch cost.

Profiling showed T3 decoding at ~30 speech tokens/s on an RTX 4090 (25 tokens = 1 s of audio), limited by
Python/kernel-launch overhead: ~30 Llama layers × dozens of tiny kernels per token. This module replaces
`T3.inference` with an equivalent loop whose per-token transformer step is captured once as a CUDA graph
and replayed:

  - static KV cache (transformers StaticCache), fixed-size buffers for the step input / position / logits
  - attention: SDPA in all layers except the three the alignment analyzer reads (LLAMA_ALIGNED_HEADS:
    layers 9, 12, 13), which keep eager attention so they return attention weights; the model-level
    config stays "eager" so masks are always materialized (no skipped causal mask)
  - the analyzer's forward hooks are replaced by hooks that only keep tensor references (no `.cpu()`
    inside the captured region); the loop copies them to CPU after each replay
  - everything after the logits (CFG mix, alignment analyzer, repetition penalty, temperature, min-p,
    top-p, multinomial sampling, EOS) is a line-by-line copy of the bundled `T3.inference` loop

Falls back to the original `T3.inference` when the prompt + max_new_tokens does not fit the cache.
"""

from __future__ import annotations

import copy

import torch


class GraphedT3Inference:
    def __init__(self, t3, max_cache_len: int = 1400, bf16: bool = False) -> None:
        from chatterbox_.models.t3.inference.alignment_stream_analyzer import (
            LLAMA_ALIGNED_HEADS,
            AlignmentStreamAnalyzer,
        )

        self.t3 = t3
        self.orig_inference = t3.inference
        self.max_cache_len = max_cache_len
        self.device = t3.device
        tfmr = t3.tfmr
        # bf16: the Llama backbone only (the card trained in BF16); embeddings, conditioning encoder and
        # speech head stay fp32. Halves the per-token weight traffic, which bounds the graphed step.
        self.dtype = torch.bfloat16 if bf16 else torch.float32
        if bf16:
            tfmr.to(torch.bfloat16)

        # Per-layer attention: SDPA everywhere except the analyzer's layers; top-level config stays eager.
        cfg = tfmr.config
        cfg._attn_implementation = "eager"
        sdpa_cfg = copy.copy(cfg)
        sdpa_cfg._attn_implementation = "sdpa"
        aligned_layers = {layer for layer, _ in LLAMA_ALIGNED_HEADS}
        for idx, layer in enumerate(tfmr.layers):
            if idx not in aligned_layers:
                layer.self_attn.config = sdpa_cfg

        self.heads = LLAMA_ALIGNED_HEADS
        self._attn_refs: list[torch.Tensor | None] = [None] * len(LLAMA_ALIGNED_HEADS)
        for k, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            tfmr.layers[layer_idx].self_attn.register_forward_hook(self._make_hook(k, head_idx))

        class _NoSpyAnalyzer(AlignmentStreamAnalyzer):
            def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):  # hooks handled above
                pass

        self._analyzer_cls = _NoSpyAnalyzer

        from transformers import StaticCache
        self.cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
        dim = cfg.hidden_size
        self.s_emb = torch.zeros(2, 1, dim, device=self.device, dtype=self.dtype)
        self.s_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        # 4D additive mask (B, 1, q=1, kv=max_cache_len), opened one slot per step outside the graph:
        # transformers' own mask builder creates a scalar tensor on the GPU each call (H2D copy), which
        # cannot be captured. A 4D mask is used as-is.
        self.min_val = torch.finfo(self.dtype).min
        self.s_mask = torch.full((2, 1, 1, max_cache_len), self.min_val, device=self.device, dtype=self.dtype)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.g_logits: torch.Tensor | None = None
        self.g_attn: list[torch.Tensor] = []

    def _make_hook(self, k: int, head_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._attn_refs[k] = output[1][0, head_idx]  # (Tq, Tkv) — reference only, no sync
        return hook

    def _forward(self, embeds: torch.Tensor, cache_position: torch.Tensor,
                 attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.t3.tfmr(inputs_embeds=embeds.to(self.dtype), attention_mask=attention_mask,
                           past_key_values=self.cache, cache_position=cache_position, use_cache=True,
                           output_attentions=False, output_hidden_states=False)
        return self.t3.speech_head(out.last_hidden_state.float())

    def _decode_step(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.graph is None:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # warm-up rewrites the same KV slot with the same values
                    self._forward(self.s_emb, self.s_pos, self.s_mask)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.g_logits = self._forward(self.s_emb, self.s_pos, self.s_mask)
            self.g_attn = list(self._attn_refs)
        self.graph.replay()
        return self.g_logits, self.g_attn

    @torch.inference_mode()
    def __call__(self, *, t3_cond, text_tokens, initial_speech_tokens=None, prepend_prompt_speech_tokens=None,
                 num_return_sequences=1, max_new_tokens=None, stop_on_eos=True, do_sample=True, temperature=0.8,
                 top_p=0.95, min_p=0.05, length_penalty=1.0, repetition_penalty=1.2, cfg_weight=0.5):
        from chatterbox_.models.t3.t3 import _ensure_BOT_EOT
        from transformers.generation.logits_process import (
            MinPLogitsWarper,
            RepetitionPenaltyLogitsProcessor,
            TopPLogitsWarper,
        )

        t3, hp = self.t3, self.t3.hp
        max_new_tokens = max_new_tokens or hp.max_speech_tokens
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        if initial_speech_tokens is None:
            initial_speech_tokens = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, len_cond = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens,
                                                   speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight)
        if embeds.size(0) != 2 or embeds.size(1) + 1 + max_new_tokens + 1 > self.max_cache_len:
            return self.orig_inference(t3_cond=t3_cond, text_tokens=text_tokens, max_new_tokens=max_new_tokens,
                                       temperature=temperature, top_p=top_p, min_p=min_p,
                                       repetition_penalty=repetition_penalty, cfg_weight=cfg_weight)

        analyzer = None
        if hp.is_multilingual:
            analyzer = self._analyzer_cls(t3.tfmr, None, text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                                          alignment_layer_idx=9, eos_idx=hp.stop_speech_token)

        bos_token = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=self.device)
        bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        prefill_len = inputs_embeds.size(1)

        self.cache.reset()
        self.s_mask.fill_(self.min_val)
        self.s_mask[..., :prefill_len] = 0.0
        logits_all = self._forward(inputs_embeds, torch.arange(prefill_len, device=self.device))
        attn = [a.float().cpu() for a in self._attn_refs]

        generated_ids = bos_token.clone()
        predicted = []
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        for i in range(max_new_tokens):
            logits_step = logits_all[:, -1, :]
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)

            if analyzer is not None:
                analyzer.last_aligned_attns = attn
                last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
                logits = analyzer.step(logits, next_token=last_token)

            ids_for_proc = generated_ids[:1, ...]
            logits = repetition_penalty_processor(ids_for_proc, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.view(-1) == hp.stop_speech_token:
                break

            next_embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            self.s_emb.copy_(torch.cat([next_embed, next_embed]))
            self.s_pos.fill_(prefill_len + i)
            self.s_mask[..., prefill_len + i] = 0.0
            logits_all, g_attn = self._decode_step()
            attn = [a.float().cpu() for a in g_attn]

        return torch.cat(predicted, dim=1)


def enable_cuda_graph(model, max_cache_len: int = 1400, bf16: bool = False) -> GraphedT3Inference:
    """Swap model.t3.inference (ChatterboxTTS / ChatterboxMultilingualTTS) for the graphed decoder."""
    graphed = GraphedT3Inference(model.t3, max_cache_len=max_cache_len, bf16=bf16)
    model.t3.inference = graphed
    return graphed


__all__ = ["GraphedT3Inference", "enable_cuda_graph"]
