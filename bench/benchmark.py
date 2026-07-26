#!/usr/bin/env python3
"""What constrained decoding costs and what it buys.

Three questions, measured separately because they have different answers:

  1. is the fast masker correct?
     trie output must be bit-identical to replaying every token. Anything less
     and the speedup is meaningless.

  2. what does masking cost per token?
     broken down by automaton state, because the answer varies by three orders
     of magnitude depending on how tight the grammar is at that point.

  3. what does it buy?
     schema validity of a real model, constrained vs not.

  python bench/benchmark.py            # masker only, no model download
  python bench/benchmark.py --model    # adds the real-model validity run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gd.automaton import JsonAutomaton, compile_schema  # noqa: E402
from gd.mask import TokenMasker, VocabTrie, constrained_generate  # noqa: E402

SCHEMAS = {
    "flat": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                       "active": {"type": "boolean"}},
        "required": ["name", "age", "active"],
    },
    "enum-heavy": {
        "type": "object",
        "properties": {"status": {"enum": ["open", "closed", "pending"]},
                       "priority": {"enum": [1, 2, 3]},
                       "owner": {"type": "string"}},
        "required": ["status", "priority", "owner"],
    },
    "nested": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "meta": {"type": "object",
                     "properties": {"score": {"type": "number"},
                                    "ok": {"type": "boolean"}},
                     "required": ["score", "ok"]},
        },
        "required": ["id", "tags", "meta"],
    },
}

# prefixes chosen to land the automaton in structurally different states
PROBES = [
    ("start", ""),
    ("after-brace", "{"),
    ("in-key", '{"na'),
    ("expect-colon", '{"name"'),
    ("in-string-value", '{"name":"ab'),
    ("in-integer", '{"name":"x","age":4'),
]


def build_vocab(tokenizer) -> list[str]:
    return [tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(i)])
            for i in range(len(tokenizer))]


def correctness_and_cost(trie: VocabTrie, root, probes) -> list[dict]:
    """Trie vs replay: identical output, and how much faster."""
    fast = TokenMasker(trie, cache=False, use_trie=True)
    slow = TokenMasker(trie, cache=False, use_trie=False)
    rows = []
    for label, prefix in probes:
        a = JsonAutomaton(root)
        if not a.feed(prefix):
            continue
        t0 = time.perf_counter(); mf = fast.mask(a); tf = time.perf_counter() - t0
        t0 = time.perf_counter(); ms = slow.mask(a); ts = time.perf_counter() - t0
        rows.append({
            "state": label,
            "allowed": int(mf.sum()),
            "allowed_pct": round(100 * mf.sum() / len(trie), 2),
            "trie_ms": round(tf * 1000, 3),
            "replay_ms": round(ts * 1000, 3),
            "speedup": round(ts / tf, 1) if tf else 0.0,
            "identical": bool(np.array_equal(mf, ms)),
        })
    return rows


class StubModel:
    """Uniform logits. Isolates masking cost from model cost.

    With a real model you can't tell whether a slow step was the mask or the
    forward pass; with this you can.
    """

    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def logits(self, ids):
        self.calls += 1
        return self.rng.standard_normal(self.vocab_size).astype(np.float32)


class StubTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def encode(self, text): return []
    def decode(self, ids): return "".join(self.vocab[i] for i in ids)


def cache_impact(trie: VocabTrie, root, vocab, max_tokens=120) -> list[dict]:
    """End-to-end generation with the cache on and off."""
    rows = []
    for cache in (False, True):
        masker = TokenMasker(trie, cache=cache, use_trie=True)
        model = StubModel(len(trie), seed=1)
        r = constrained_generate(model, StubTokenizer(vocab), "", root, masker,
                                 max_tokens=max_tokens, temperature=1.0, seed=1)
        rows.append({
            "cache": cache,
            "complete": r.complete,
            "steps": r.steps,
            "mask_ms_total": round(r.mask_seconds * 1000, 1),
            "mask_ms_per_token": round(r.mask_seconds * 1000 / max(r.steps, 1), 3),
            "masks_computed": masker.stats.computed,
            "cache_hit_rate": round(masker.stats.hit_rate, 3),
            "mean_allowed": round(r.mean_allowed, 1),
        })
    return rows


def schema_valid(text: str, schema: dict) -> bool:
    """Conformance, not just parseability.

    Checking only json.loads flatters the baseline badly: a model that emits
    `5` or `"hello"` produces perfectly valid JSON that satisfies none of the
    schema. The claim is about conformance, so that is what gets measured.
    """
    import jsonschema

    try:
        jsonschema.validate(json.loads(text), schema)
        return True
    except Exception:
        return False


def real_model_validity(schema_name: str, schema: dict, n: int, max_tokens: int) -> dict:
    """Does a real model emit schema-conforming JSON, with and without the mask?"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name)
    hf.eval()
    vocab = build_vocab(tok)
    trie = VocabTrie(trie_vocab := vocab)
    root = compile_schema(schema)

    class HF:
        def __init__(self):
            self.calls = 0

        def logits(self, ids):
            self.calls += 1
            with torch.no_grad():
                x = torch.tensor([ids or [tok.eos_token_id]], dtype=torch.long)
                return hf(x).logits[0, -1].numpy()

    prompt = (f"Produce a JSON object matching this schema:\n"
              f"{json.dumps(schema)}\nJSON:\n")

    # constrained
    ok_c = 0
    times_c = []
    for i in range(n):
        masker = TokenMasker(trie, cache=True, use_trie=True)
        r = constrained_generate(HF(), tok, prompt, root, masker,
                                 max_tokens=max_tokens, temperature=0.8, seed=i,
                                 closing_budget=max(20, max_tokens // 3))
        times_c.append(r.as_dict())
        if r.complete and schema_valid(r.text, schema):
            ok_c += 1

    # unconstrained: same prompt, same model, no mask
    ok_u = 0
    for i in range(n):
        torch.manual_seed(i)
        ids = tok.encode(prompt, return_tensors="pt")
        out = hf.generate(ids, max_new_tokens=max_tokens, do_sample=True,
                          temperature=0.8, top_p=0.95,
                          pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        # give the baseline every chance: try each line, and the whole thing
        for cand in [text.strip(), *text.strip().splitlines()]:
            if schema_valid(cand, schema):
                ok_u += 1
                break

    overhead = sum(t["masking_overhead"] for t in times_c) / max(len(times_c), 1)
    return {
        "schema": schema_name,
        "samples": n,
        "constrained_valid": ok_c,
        "constrained_rate": round(ok_c / n, 3),
        "unconstrained_valid": ok_u,
        "unconstrained_rate": round(ok_u / n, 3),
        "mean_masking_overhead": round(overhead, 4),
    }


def table(rows, cols):
    head = "".join(n[:w].ljust(w + 1) if i == 0 else n[:w].rjust(w + 1)
                   for i, (n, w) in enumerate(cols))
    out = [head, "-" * len(head)]
    for r in rows:
        line = ""
        for i, (n, w) in enumerate(cols):
            v = r.get(n, "-")
            line += str(v).ljust(w + 1) if i == 0 else str(v).rjust(w + 1)
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="also run the real-model validity test")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    vocab = build_vocab(tok)
    t0 = time.perf_counter()
    trie = VocabTrie(vocab)
    build_s = time.perf_counter() - t0
    print(f"vocabulary {len(trie)} tokens, trie built in {build_s * 1000:.0f} ms "
          f"(once per tokenizer, reused for every schema)\n")

    result = {"vocab_size": len(trie), "trie_build_seconds": round(build_s, 4)}

    root = compile_schema(SCHEMAS["flat"])
    print("mask correctness and cost, by automaton state (schema: flat)")
    rows = correctness_and_cost(trie, root, PROBES)
    print(table(rows, [("state", 18), ("allowed", 9), ("allowed_pct", 13),
                       ("trie_ms", 10), ("replay_ms", 11), ("speedup", 9),
                       ("identical", 11)]))
    result["masking"] = rows
    assert all(r["identical"] for r in rows), "trie masker disagrees with replay"
    print("\n  every state: trie output is bit-identical to replaying all "
          f"{len(trie)} tokens.")
    print("  speedup tracks how much the grammar prunes. a tight state kills")
    print("  most of the trie in one comparison; inside a string almost every")
    print("  token is legal, so there is nothing to prune and the trie barely wins.\n")

    print("cache impact, generating one document end to end (stub model)")
    rows = cache_impact(trie, root, vocab)
    print(table(rows, [("cache", 7), ("steps", 7), ("complete", 10),
                       ("mask_ms_total", 15), ("mask_ms_per_token", 19),
                       ("masks_computed", 16), ("cache_hit_rate", 16)]))
    result["cache"] = rows
    if len(rows) == 2 and rows[0]["mask_ms_total"] and rows[1]["mask_ms_total"]:
        print(f"\n  cache speedup: {rows[0]['mask_ms_total'] / rows[1]['mask_ms_total']:.1f}x")
        print("  the state space of a schema is tiny and the same states recur")
        print("  constantly, so the mask is computed once per state, not per token.\n")

    print("schema coverage")
    cov = []
    for name, schema in SCHEMAS.items():
        r = compile_schema(schema)
        masker = TokenMasker(trie, cache=True)
        gen = constrained_generate(StubModel(len(trie), seed=3), StubTokenizer(vocab),
                                   "", r, masker, max_tokens=200, temperature=1.0,
                                   seed=3, closing_budget=60)
        valid = False
        if gen.complete:
            try:
                json.loads(gen.text)
                valid = True
            except json.JSONDecodeError:
                pass
        cov.append({"schema": name, "complete": gen.complete, "parses": valid,
                    "steps": gen.steps,
                    "mask_ms_per_token": round(gen.mask_seconds * 1000 / max(gen.steps, 1), 3),
                    "sample": gen.text[:60]})
    print(table(cov, [("schema", 12), ("complete", 10), ("parses", 8), ("steps", 7),
                      ("mask_ms_per_token", 19), ("sample", 62)]))
    result["coverage"] = cov
    print("\n  a stub model emits uniformly random logits, so the *content* is")
    print("  noise. it still parses, because validity comes from the mask rather")
    print("  than from the model behaving.")
    print("  note the mask alone guarantees a valid *prefix*, not termination:")
    print("  inside a string every character is legal, so a model that never")
    print("  picks the closing quote runs to the limit. closing_budget narrows")
    print("  the mask to structure-closing characters near the budget.\n")

    if args.model:
        print("real model: distilgpt2, constrained vs unconstrained")
        rows = [real_model_validity(n, s, args.samples, args.max_tokens)
                for n, s in SCHEMAS.items()]
        print(table(rows, [("schema", 12), ("samples", 9), ("constrained_rate", 18),
                           ("unconstrained_rate", 20), ("mean_masking_overhead", 23)]))
        result["real_model"] = rows

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
