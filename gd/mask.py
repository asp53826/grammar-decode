"""Turning a character automaton into a token mask.

The automaton says which *characters* are legal. A model emits *tokens*, and a
GPT-2 token is an arbitrary byte string like `":"` or ` {"` or `true`. So the
real question at each step is:

    which of the 50,257 tokens, when appended, keep the output on a path the
    grammar can still complete?

The naive answer is to replay every token through a cloned automaton on every
step. That is 50,257 clones per token generated, and it is unusably slow — the
benchmark measures exactly how slow.

Three things make it fast:

  1. a prefix trie over the vocabulary
     Tokens sharing a prefix share a fate. If no token may start with `z`, the
     entire `z` subtree is pruned in one comparison instead of once per token.

  2. a mask cache keyed on automaton state
     The state space of a schema is tiny (tens of states) while the vocabulary
     is huge. The same state recurs constantly — every string character, every
     digit — so the mask is computed once per *state*, not once per token.

  3. incremental descent
     Walking the trie and the automaton together means a token is validated
     character by character with early exit, rather than cloning and replaying.

The cache is what actually wins, and the benchmark separates the three so the
contribution of each is visible rather than asserted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .automaton import JsonAutomaton, Node


class TrieNode:
    __slots__ = ("children", "token_ids", "terminal")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        # tokens that end exactly here
        self.token_ids: list[int] = []
        self.terminal = False


class VocabTrie:
    """Prefix trie over the decoded vocabulary.

    Built once per tokenizer and reused for every schema and every request,
    because it depends only on the vocabulary.
    """

    def __init__(self, vocab: list[str], skip: set[int] | None = None):
        self.vocab = vocab
        self.size = len(vocab)
        self.root = TrieNode()
        self.skipped = set(skip or ())
        self.unusable = 0

        for tid, text in enumerate(vocab):
            if tid in self.skipped or not text:
                self.unusable += 1
                continue
            node = self.root
            for ch in text:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = node.children[ch] = TrieNode()
                node = nxt
            node.token_ids.append(tid)
            node.terminal = True

    def __len__(self) -> int:
        return self.size


@dataclass
class MaskStats:
    computed: int = 0        # masks actually built
    cache_hits: int = 0
    tokens_examined: int = 0
    seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.computed + self.cache_hits
        return self.cache_hits / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "masks_computed": self.computed,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.hit_rate, 4),
            "tokens_examined": self.tokens_examined,
            "mask_seconds": round(self.seconds, 4),
        }


class TokenMasker:
    """Produces a boolean mask over the vocabulary for an automaton state."""

    def __init__(self, trie: VocabTrie, cache: bool = True, use_trie: bool = True):
        self.trie = trie
        self.enable_cache = cache
        self.use_trie = use_trie
        self._cache: dict[tuple, np.ndarray] = {}
        self.stats = MaskStats()

    def clear_cache(self) -> None:
        self._cache.clear()

    def mask(self, automaton: JsonAutomaton) -> np.ndarray:
        """Boolean array, True where the token is legal next."""
        t0 = time.perf_counter()
        key = automaton.state_key() if self.enable_cache else None

        if key is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                self.stats.seconds += time.perf_counter() - t0
                return hit

        allowed = (self._mask_via_trie(automaton) if self.use_trie
                   else self._mask_via_replay(automaton))

        self.stats.computed += 1
        if key is not None:
            self._cache[key] = allowed
        self.stats.seconds += time.perf_counter() - t0
        return allowed

    # ------------------------------------------------------------ fast path

    def _mask_via_trie(self, automaton: JsonAutomaton) -> np.ndarray:
        out = np.zeros(self.trie.size, dtype=bool)
        # walk the trie and the automaton together; a dead branch prunes every
        # token beneath it at once
        self._descend(self.trie.root, automaton.clone(), out)
        return out

    def _descend(self, node: TrieNode, state: JsonAutomaton, out: np.ndarray) -> None:
        legal = state.allowed()
        if not legal:
            return
        for ch, child in node.children.items():
            if ch not in legal:
                continue
            self.stats.tokens_examined += 1
            nxt = state.clone()
            if not nxt.advance(ch):
                continue
            if child.terminal:
                # a token ending here is legal as long as the grammar is still
                # satisfiable, which advance() already guaranteed
                for tid in child.token_ids:
                    out[tid] = True
            if child.children:
                self._descend(child, nxt, out)

    # ------------------------------------------------------------ slow path

    def _mask_via_replay(self, automaton: JsonAutomaton) -> np.ndarray:
        """The obvious implementation: try every token independently.

        Kept because it is obviously correct, which makes it the reference the
        fast path is tested against — and because the benchmark needs something
        to be faster *than*.
        """
        out = np.zeros(self.trie.size, dtype=bool)
        for tid, text in enumerate(self.trie.vocab):
            if tid in self.trie.skipped or not text:
                continue
            self.stats.tokens_examined += 1
            probe = automaton.clone()
            if probe.feed(text):
                out[tid] = True
        return out

    # ------------------------------------------------------------ helpers

    def apply(self, logits: np.ndarray, automaton: JsonAutomaton) -> np.ndarray:
        """Set disallowed logits to -inf. Returns a new array."""
        allowed = self.mask(automaton)
        out = logits.astype(np.float32, copy=True)
        out[~allowed] = -np.inf
        return out

    def num_allowed(self, automaton: JsonAutomaton) -> int:
        return int(self.mask(automaton).sum())

    def mask_for_chars(self, automaton: JsonAutomaton,
                       first_chars: frozenset[str]) -> np.ndarray:
        """Legal tokens restricted to those starting with one of `first_chars`.

        Used to apply closing pressure near the token budget without ever
        allowing something the grammar rejects: this is a subset of mask(),
        never a superset.
        """
        out = np.zeros(self.trie.size, dtype=bool)
        for ch in first_chars:
            child = self.trie.root.children.get(ch)
            if child is None:
                continue
            nxt = automaton.clone()
            if not nxt.advance(ch):
                continue
            for tid in child.token_ids:
                out[tid] = True
            if child.children:
                self._descend(child, nxt, out)
        return out


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    complete: bool = False
    steps: int = 0
    seconds: float = 0.0
    mask_seconds: float = 0.0
    model_seconds: float = 0.0
    mean_allowed: float = 0.0
    stop_reason: str = ""

    @property
    def overhead_fraction(self) -> float:
        total = self.mask_seconds + self.model_seconds
        return self.mask_seconds / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "complete": self.complete,
            "steps": self.steps,
            "seconds": round(self.seconds, 4),
            "mask_seconds": round(self.mask_seconds, 4),
            "model_seconds": round(self.model_seconds, 4),
            "masking_overhead": round(self.overhead_fraction, 4),
            "mean_allowed_tokens": round(self.mean_allowed, 1),
            "stop_reason": self.stop_reason,
        }


def closing_chars(automaton: JsonAutomaton) -> frozenset[str]:
    """Legal characters that make structural progress toward finishing.

    The comma belongs here as much as the brackets do. Without it a model deep
    inside a number can never be pushed onward — no bracket is legal there, so
    the only way out is the separator, and omitting it means closing pressure
    silently does nothing exactly where it is most needed.
    """
    return frozenset(c for c in automaton.allowed() if c in '"}],')


def constrained_generate(model, tokenizer, prompt: str, root: Node,
                         masker: TokenMasker, max_tokens: int = 200,
                         temperature: float = 0.0, seed: int = 0,
                         closing_budget: int = 0) -> GenerationResult:
    """Greedy or sampled decoding with the mask applied before selection.

    `model` only has to expose `logits(token_ids) -> np.ndarray`, so this works
    against a real transformer or a stub. The point being demonstrated is the
    masking, not the model.

    A mask guarantees the output stays on a *valid path*. It cannot make the
    model choose to finish: inside a string every character is legal, so a
    model that never picks the closing quote runs to max_tokens holding a
    perfectly valid but unterminated prefix. `closing_budget` is the fix —
    within that many tokens of the limit, the mask narrows to characters that
    close structures, so generation lands on a complete document instead of
    being cut off mid-string.
    """
    rng = np.random.default_rng(seed)
    automaton = JsonAutomaton(root)
    ids = list(tokenizer.encode(prompt))
    generated: list[int] = []
    allowed_counts: list[int] = []
    mask_s = model_s = 0.0
    t0 = time.perf_counter()
    stop = "length"

    for step in range(max_tokens):
        if automaton.is_complete():
            stop = "grammar_complete"
            break

        m0 = time.perf_counter()
        allowed = masker.mask(automaton)
        remaining = max_tokens - step
        if closing_budget and remaining <= closing_budget:
            closers = closing_chars(automaton)
            if closers:
                narrowed = masker.mask_for_chars(automaton, closers)
                if narrowed.any():
                    allowed = narrowed
        mask_s += time.perf_counter() - m0
        n_allowed = int(allowed.sum())
        allowed_counts.append(n_allowed)

        if n_allowed == 0:
            # the grammar admits nothing, which means either the schema is
            # unsatisfiable or the vocabulary can't express it. either way it's
            # a bug to surface, not to paper over.
            stop = "no_legal_token"
            break

        g0 = time.perf_counter()
        logits = model.logits(ids + generated)
        model_s += time.perf_counter() - g0

        logits = logits.astype(np.float32, copy=True)
        logits[~allowed] = -np.inf

        if temperature <= 0:
            tid = int(np.argmax(logits))
        else:
            z = logits / temperature
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            tid = int(rng.choice(len(p), p=p))

        generated.append(tid)
        text = tokenizer.decode([tid])
        if not automaton.feed(text):
            # can't happen if the mask is right, so it's an assertion about the
            # masker rather than a runtime condition
            stop = "mask_violation"
            break

    return GenerationResult(
        text=automaton.text,
        token_ids=generated,
        complete=automaton.is_complete(),
        steps=len(generated),
        seconds=time.perf_counter() - t0,
        mask_seconds=mask_s,
        model_seconds=model_s,
        mean_allowed=sum(allowed_counts) / len(allowed_counts) if allowed_counts else 0.0,
        stop_reason=stop,
    )
