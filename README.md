# grammar-decode

Constrained decoding: compile a JSON Schema into a character automaton, turn
that into a token mask over a 50k vocabulary, and delete illegal tokens from
the logits before sampling. The model cannot emit invalid output because the
invalid options are not on the ballot.

```bash
pip install -r requirements-dev.txt
pytest                              # 50 tests
python bench/benchmark.py           # masking cost, no model download
python bench/benchmark.py --model   # adds real-model conformance
```

## Results

**distilgpt2, 12 samples per schema, validated against the schema** (not merely
parsed as JSON — see the note below on why that distinction matters):

| schema | constrained | unconstrained |
|---|---|---|
| flat (3 scalars) | **91.7%** | 0% |
| enum-heavy | **100%** | 0% |
| nested (array + sub-object) | **100%** | 0% |

Masking overhead is ~21-28% of step time against a *tiny* model on CPU. That
fraction shrinks as the model grows, because the mask cost is fixed per step
while the forward pass isn't.

The 91.7% is honest: one run of twelve hit the token limit mid-string. The mask
guarantees a valid *path*, not termination — see below.

### Masking cost by automaton state

The interesting result. Trie-walking vs replaying all 50,257 tokens:

| state | tokens allowed | trie | replay | speedup |
|---|---|---|---|---|
| start | 2 (0.0%) | 0.08 ms | 42.7 ms | **521x** |
| in-key | 2 (0.0%) | 0.07 ms | 80.7 ms | **1203x** |
| expect-colon | 2 (0.0%) | 0.05 ms | 44.5 ms | **876x** |
| in-integer | 999 (2.0%) | 2.10 ms | 85.7 ms | **41x** |
| in-string-value | 49,275 (98.1%) | 295.7 ms | 441.4 ms | **1.5x** |

**Speedup tracks how much the grammar prunes, and nothing else.** A tight state
kills most of the trie in one comparison. Inside a string almost every token is
legal, so there is nothing to prune and the trie barely beats brute force.

That's why the cache is what actually matters:

| | mask time / token | masks computed | hit rate |
|---|---|---|---|
| no cache | 313 ms | 120 | 0% |
| cache | **2.4 ms** | **7** | 94% |

**132x.** A schema's state space is tiny — 7 distinct states across a 120-token
document — while the vocabulary is huge. Compute the mask once per *state*, not
once per token.

## How it works

```
JSON Schema  →  Node tree  →  character automaton  →  vocab trie walk  →  bool mask  →  logits[~mask] = -inf
```

The automaton answers one question: *given what has been emitted, which
characters may come next?* That's cheap because JSON is regular once the schema
fixes key order and value types. The stack only tracks nesting depth, and the
schema bounds that.

Turning characters into tokens is the hard part. A GPT-2 token is an arbitrary
byte string — `":"`, `` `":true` ``, `` `"},` `` — so masking means asking, for
each of 50,257 tokens, whether appending it keeps the output on a completable
path. Three things make that fast:

1. **A prefix trie over the vocabulary.** Tokens sharing a prefix share a fate;
   if no token may start with `z`, the whole subtree dies in one comparison.
2. **A mask cache keyed on automaton state.** The 132x above.
3. **Joint descent.** Walk trie and automaton together with early exit, instead
   of cloning and replaying each token.

The slow replay path is kept, because it's obviously correct and is therefore
what the fast path is tested against.

## Five bugs, and what they have in common

Every one of these produced output that *parsed as JSON* and was wrong anyway.
That's the failure mode of an over-permissive mask: nothing crashes, and the
error surfaces at a schema check somewhere downstream.

**1. Objects could close with required keys missing.** `_terminators()` returned
`{",", "}"}` for any object parent without checking `key_index`, so after two of
three keys the mask allowed `}`. Output: `{"name":"...","age":1}` — valid JSON,
missing `active`.

**2. Leading zeros.** Emitted `{"age":00000000106}`. JSON's grammar is
`int = "0" / (digit1-9 *DIGIT)`; I allowed digits freely after the first.

**3. Trailing commas in arrays.** `["data",]`, because "start of array" and
"just after a comma" shared a state, so `]` stayed legal after the separator.

**4. Exponents under `"type": "integer"`.** `1e5` is a float to `json.loads`,
and `9e999999` is `inf` — valid JSON syntax, not a number. Exponents now belong
to `number` only, and are capped at 3 digits.

**5. Two unsound cache keys — same root cause, twice.**

```python
min(f.item_count, 2)     # collides "2 elements" with "3" when maxItems == 3
min(f.seen_digits, 2)    # collides "2 exponent digits" with "3" once capped
```

Both let the cache hand back a mask built for a *different* state. The second
one appeared the moment I added the exponent cap — I fixed the first, added a
new threshold, and reintroduced the identical bug in a new place.

The lesson, which is now a comment in the source: **every clamp in a cache key
has to be justified against every threshold the transition function tests.** A
clamp is an assertion that two states are interchangeable, and adding a
threshold silently invalidates it.

All five became regression tests. The one that caught #5 is the one worth
having:

```python
def test_generation_never_violates_its_own_mask(trie):
    """If this fires, mask() and advance() disagree — a bug in the masker
    rather than a runtime condition."""
    assert r.stop_reason != "mask_violation"
```

## A mask guarantees validity, not termination

Inside a string every character is legal, so a model that never picks the
closing quote runs to `max_tokens` holding a perfectly valid, unterminated
prefix. No amount of masking fixes that — the constraint says what *may* come
next, never what *must*.

`closing_budget` handles it: within N tokens of the limit, the mask narrows to
characters that close structures. That set has to include the comma, not just
`"`/`}`/`]` — a model deep inside a number can't be closed by any bracket, and
omitting the comma made closing pressure silently do nothing exactly where it
was most needed.

Narrowing is always a **subset** of the legal mask, never a superset. There's a
test.

## Measuring conformance, not parseability

My first version scored the baseline with `json.loads`. That flatters it badly:
a model emitting `5` or `"hello"` produces valid JSON satisfying none of the
schema. Switching to `jsonschema.validate` moved the unconstrained baseline
from 25-50% to **0%** across all three schemas.

Worth stating plainly: the baseline is 0% because distilgpt2 is an 82M-parameter
model that cannot follow a schema instruction at all. Against a frontier model
the honest comparison is roughly 100% vs ~85%, and the gap is a tail-risk
argument rather than a night-and-day one.

## Being straight about what this is

- **Schema subset**, not full JSON Schema: object/array/string/integer/number/
  boolean/null/enum/const, plus `required`, `minItems`, `maxItems`. Unsupported
  keywords **raise** rather than being ignored — a constraint that silently does
  nothing is worse than one that errors.
- **Required keys are emitted in schema order** and optional keys aren't emitted
  at all. That's what keeps the grammar regular and the transitions O(1). Real
  implementations handle arbitrary key order by tracking a set of live states.
- **No `pattern`, no `minLength`, no `$ref`, no union types.** A regex would
  need its own automaton composed with this one; `$ref` needs cycle detection.
- **Python, and the mask is a numpy bool array.** A production implementation
  builds this on GPU alongside the logits. The per-state costs here are real
  but the constant factor isn't.
- **distilgpt2** is the test model because it downloads in seconds. It makes the
  baseline look terrible for reasons that are about the model, not the method.
- **Not integrated with a server.** The natural next step is wiring it into
  [vllm-lite](https://github.com/asp53826/vllm-lite)'s sampler, where the mask
  would apply per-sequence inside the existing batch loop.

## Tests

50 tests. The three that carry the weight:

- **`test_trie_mask_equals_replay_mask`** — fast path bit-identical to brute
  force across 13 prefixes. Without this the speedup is just a faster way to be
  wrong.
- **`test_every_masked_token_is_actually_legal`** — takes every token the mask
  permits and confirms the automaton accepts it. Guards the dangerous direction:
  too permissive.
- **`test_random_model_still_produces_conforming_output`** — uniform random
  logits, 3 schemas × 5 seeds, output must validate. Validity comes from the
  mask, not from the model cooperating.

Plus one regression test per bug above, each labelled with what it caught.

## Layout

```
gd/automaton.py    schema → Node tree → character automaton, state keys
gd/mask.py         vocab trie, cached masker, replay reference, generation loop
bench/benchmark.py per-state cost, cache impact, real-model conformance
tests/             50 tests
```
