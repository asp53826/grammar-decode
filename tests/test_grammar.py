"""Tests for the automaton and the masker.

The load-bearing one is `test_trie_mask_equals_replay_mask`: the fast path must
agree with the obviously-correct slow path everywhere, or the speedup is just a
faster way to be wrong.

Four tests here exist because they caught real bugs, and each is marked. A mask
that is too *strict* shows up immediately as garbled output. A mask that is too
*permissive* produces text that looks fine and fails a schema check downstream,
which is why those get their own tests.
"""

import json

import jsonschema
import numpy as np
import pytest

from gd.automaton import JsonAutomaton, SchemaError, compile_schema
from gd.mask import TokenMasker, VocabTrie, closing_chars, constrained_generate

FLAT = {"type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                       "active": {"type": "boolean"}},
        "required": ["name", "age", "active"]}

NESTED = {"type": "object",
          "properties": {"id": {"type": "integer"},
                         "tags": {"type": "array", "items": {"type": "string"},
                                  "maxItems": 3},
                         "meta": {"type": "object",
                                  "properties": {"score": {"type": "number"},
                                                 "ok": {"type": "boolean"}},
                                  "required": ["score", "ok"]}},
          "required": ["id", "tags", "meta"]}

ENUMS = {"type": "object",
         "properties": {"status": {"enum": ["open", "closed", "pending"]},
                        "n": {"enum": [1, 2, 3]}},
         "required": ["status", "n"]}


def accepts(schema, text):
    a = JsonAutomaton(compile_schema(schema))
    return a.feed(text) and a.is_complete()


# ---------------------------------------------------------------- acceptance

@pytest.mark.parametrize("doc", [
    '{"name":"ada","age":36,"active":true}',
    '{"name":"","age":0,"active":false}',
    '{"name":"a b c","age":-5,"active":true}',
    '{"name":"quote\\"inside","age":10,"active":false}',
])
def test_accepts_valid_documents(doc):
    assert accepts(FLAT, doc)
    jsonschema.validate(json.loads(doc), FLAT)


@pytest.mark.parametrize("doc", [
    '{"nome":"x","age":1,"active":true}',      # wrong key
    '{"name":"x","age":"1","active":true}',    # string where integer required
    '{"name":"x","active":true,"age":1}',      # keys out of order
    '{"name":"x","age":1}',                    # missing required key
    '{"age":1,"name":"x","active":true}',      # reordered
])
def test_rejects_invalid_documents(doc):
    assert not accepts(FLAT, doc)


def test_nested_and_enums():
    assert accepts(NESTED, '{"id":1,"tags":["a","b"],"meta":{"score":1.5,"ok":true}}')
    assert accepts(NESTED, '{"id":0,"tags":[],"meta":{"score":-0.5,"ok":false}}')
    assert accepts(ENUMS, '{"status":"pending","n":2}')
    assert not accepts(ENUMS, '{"status":"unknown","n":2}')
    assert not accepts(ENUMS, '{"status":"open","n":9}')


def test_max_items_is_enforced():
    assert accepts(NESTED, '{"id":1,"tags":["a","b","c"],"meta":{"score":1,"ok":true}}')
    assert not accepts(NESTED,
                       '{"id":1,"tags":["a","b","c","d"],"meta":{"score":1,"ok":true}}')


# ------------------------------------------- bugs found by the real-model run

def test_object_cannot_close_while_keys_remain():
    """REGRESSION: _terminators returned {',', '}'} for any object parent
    without checking key_index, so the mask allowed closing after two of three
    keys. The result parsed as JSON and silently failed schema validation."""
    a = JsonAutomaton(compile_schema(FLAT))
    assert a.feed('{"name":"x","age":1')
    assert "}" not in a.allowed(), "closed early with 'active' still required"
    assert "," in a.allowed()

    a = JsonAutomaton(compile_schema(FLAT))
    a.feed('{"name":"x","age":1,"active":true')
    assert "}" in a.allowed()
    assert not accepts(FLAT, '{"name":"x","age":1}')


def test_leading_zeros_are_rejected():
    """REGRESSION: emitted 00000000106, which this automaton accepted and
    json.loads did not."""
    a = JsonAutomaton(compile_schema(FLAT))
    a.feed('{"name":"x","age":0')
    assert not (a.allowed() & set("0123456789"))
    assert not accepts(FLAT, '{"name":"x","age":007,"active":true}')
    assert accepts(FLAT, '{"name":"x","age":0,"active":true}')
    assert accepts(FLAT, '{"name":"x","age":10,"active":true}')


def test_no_trailing_comma_in_arrays():
    """REGRESSION: produced ["data",] because 'after a comma' and 'start of
    array' shared a state, so ']' stayed legal after the separator."""
    root = compile_schema({"type": "array", "items": {"type": "string"}})
    a = JsonAutomaton(root)
    a.feed('["a",')
    assert "]" not in a.allowed(), "trailing comma would be allowed"
    assert '"' in a.allowed()
    b = JsonAutomaton(root)
    assert not b.feed('["a",]')


def test_closing_pressure_includes_the_comma():
    """REGRESSION: closing_chars only looked at " } ], so a model deep inside a
    number could never be pushed onward — no bracket is legal there."""
    a = JsonAutomaton(compile_schema(FLAT))
    a.feed('{"name":"x","age":12')
    assert "," in closing_chars(a), "no way out of a number without the comma"


# ---------------------------------------------------------------- schema errors

def test_unsupported_keywords_are_rejected_loudly():
    """Silently ignoring a constraint is worse than refusing it."""
    with pytest.raises(SchemaError, match="unsupported schema keyword"):
        compile_schema({"type": "string", "minLength": 5})
    with pytest.raises(SchemaError, match="union types"):
        compile_schema({"type": ["string", "null"]})
    with pytest.raises(SchemaError):
        compile_schema({"type": "object"})           # no properties
    with pytest.raises(SchemaError):
        compile_schema({"type": "array"})            # no items
    with pytest.raises(SchemaError):
        compile_schema({"enum": []})


# ---------------------------------------------------------------- masking

@pytest.fixture(scope="module")
def trie():
    # a small hand-made vocabulary keeps the differential test fast while still
    # containing the awkward cases: multi-character tokens that span structure
    toks = ['{', '}', '[', ']', '"', ':', ',', 'a', 'b', 'x', 'y', '0', '1', '2',
            '9', 'true', 'false', 'null', 'name', 'age', 'active', '":', '":"',
            '","', '"}', '":true', '":false', ':"', '{"', '"],', 'e', '-', '.',
            'id', 'tags', 'meta', 'score', 'ok', 'open', 'closed', 'pending']
    return VocabTrie(toks)


PREFIXES = ["", "{", '{"', '{"name', '{"name"', '{"name":', '{"name":"',
            '{"name":"ab', '{"name":"ab"', '{"name":"ab",', '{"name":"ab","age":',
            '{"name":"ab","age":1', '{"name":"ab","age":1,"active":']


@pytest.mark.parametrize("prefix", PREFIXES)
def test_trie_mask_equals_replay_mask(trie, prefix):
    """The whole speedup rests on this. Fast and slow paths must agree exactly."""
    root = compile_schema(FLAT)
    a = JsonAutomaton(root)
    if not a.feed(prefix):
        pytest.skip(f"prefix {prefix!r} not reachable")
    fast = TokenMasker(trie, cache=False, use_trie=True).mask(a)
    slow = TokenMasker(trie, cache=False, use_trie=False).mask(a)
    assert np.array_equal(fast, slow), (
        f"differ on {sorted(np.flatnonzero(fast ^ slow))}")


def test_cache_does_not_change_the_mask(trie):
    root = compile_schema(NESTED)
    cached = TokenMasker(trie, cache=True)
    plain = TokenMasker(trie, cache=False)
    a = JsonAutomaton(root)
    for ch in '{"id":1,"tags":["a"':
        assert np.array_equal(cached.mask(a), plain.mask(a))
        a.advance(ch)
    assert cached.stats.cache_hits > 0, "nothing was reused, so nothing was tested"


def test_every_masked_token_is_actually_legal(trie):
    """A mask that is too permissive is the dangerous direction. Take every
    token the mask allows and confirm the automaton really accepts it."""
    root = compile_schema(FLAT)
    masker = TokenMasker(trie, cache=False)
    a = JsonAutomaton(root)
    for prefix in ("", "{", '{"name":"', '{"name":"x","age":'):
        st = JsonAutomaton(root)
        st.feed(prefix)
        allowed = masker.mask(st)
        for tid in np.flatnonzero(allowed):
            probe = st.clone()
            assert probe.feed(trie.vocab[tid]), (
                f"mask allowed {trie.vocab[tid]!r} after {prefix!r} but the "
                f"automaton rejects it")


def test_mask_for_chars_is_a_subset(trie):
    root = compile_schema(FLAT)
    masker = TokenMasker(trie, cache=False)
    a = JsonAutomaton(root)
    a.feed('{"name":"x","age":1')
    full = masker.mask(a)
    narrow = masker.mask_for_chars(a, closing_chars(a))
    assert narrow.sum() > 0
    # closing pressure must never widen what the grammar permits
    assert np.array_equal(narrow & full, narrow)


# ---------------------------------------------------------------- generation

class UniformModel:
    """Random logits. The output is nonsense; that's the point — validity has
    to come from the mask, not from the model cooperating."""

    def __init__(self, size, seed=0):
        self.size = size
        self.rng = np.random.default_rng(seed)

    def logits(self, ids):
        return self.rng.standard_normal(self.size).astype(np.float32)


class ListTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text):
        return []

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


@pytest.mark.parametrize("schema", [FLAT, NESTED, ENUMS])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_model_still_produces_conforming_output(trie, schema, seed):
    """The headline property, across schemas and seeds."""
    root = compile_schema(schema)
    masker = TokenMasker(trie, cache=True)
    r = constrained_generate(UniformModel(len(trie), seed), ListTokenizer(trie.vocab),
                             "", root, masker, max_tokens=400, temperature=1.0,
                             seed=seed, closing_budget=150)
    assert r.stop_reason != "mask_violation", "the mask allowed an illegal token"
    if r.complete:
        jsonschema.validate(json.loads(r.text), schema)


def test_generation_never_violates_its_own_mask(trie):
    """If this ever fires, mask() and advance() disagree, which is a bug in the
    masker rather than a runtime condition."""
    for seed in range(8):
        masker = TokenMasker(trie, cache=True)
        r = constrained_generate(UniformModel(len(trie), seed),
                                 ListTokenizer(trie.vocab), "", compile_schema(NESTED),
                                 masker, max_tokens=300, temperature=1.0, seed=seed,
                                 closing_budget=100)
        assert r.stop_reason != "mask_violation"


def test_greedy_generation_is_deterministic(trie):
    root = compile_schema(FLAT)
    outs = set()
    for _ in range(3):
        masker = TokenMasker(trie, cache=True)
        r = constrained_generate(UniformModel(len(trie), 7), ListTokenizer(trie.vocab),
                                 "", root, masker, max_tokens=200, temperature=0.0)
        outs.add(r.text)
    assert len(outs) == 1


def test_state_key_collapses_equivalent_states(trie):
    """The cache only works if states that permit the same characters hash the
    same. Two different string positions should share a key."""
    root = compile_schema(FLAT)
    a = JsonAutomaton(root); a.feed('{"name":"ab')
    b = JsonAutomaton(root); b.feed('{"name":"xyz')
    assert a.state_key() == b.state_key()

    c = JsonAutomaton(root); c.feed('{"name":"ab"')
    assert a.state_key() != c.state_key()
