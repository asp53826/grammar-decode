"""JSON Schema compiled to a character-level automaton.

Constrained decoding works by deleting invalid tokens from the logits before
sampling, which means at every step you need the answer to: "given what has
been emitted so far, which of the 50,257 tokens could legally come next?"

That question is only cheap if the grammar is a state machine. So a schema
compiles down to states, each of which knows which *characters* may follow.
Token-level masking is then built on top of that in mask.py.

Why a hand-written automaton rather than a general CFG parser: JSON's grammar
is regular once the schema fixes the key order and the value types, and a
regular language gives O(1) transitions. A general parser would need the whole
stack on every token, which is the difference between microseconds and
milliseconds per step, and this runs once per generated token.

The stack here is only for nesting depth (objects inside arrays inside
objects), which is genuinely context-free — but it's a *bounded* stack because
the schema fixes the maximum depth, so it stays cheap.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from typing import Iterator

DIGITS = frozenset(string.digits)
WS = frozenset(" \t\n\r")
HEX = frozenset(string.hexdigits)
# every character a JSON string may contain unescaped
STRING_CHARS = frozenset(chr(c) for c in range(0x20, 0x7F)) - {'"', "\\"}
ESCAPES = frozenset('"\\/bfnrtu')

# Longest exponent accepted, so 9e999999 (which json.loads turns into inf)
# can't be produced.
MAX_EXPONENT_DIGITS = 3

# How far seen_digits may be clamped in the cache key. Every clamp in a state
# key has to be justified against every threshold the transition function
# tests, or the cache hands back a mask for a *different* state. Two separate
# bugs came from getting this wrong — see the README.
DIGIT_STATE_CAP = MAX_EXPONENT_DIGITS + 1


class SchemaError(ValueError):
    pass


@dataclass
class Node:
    """One node of the compiled schema tree."""

    kind: str  # object | array | string | integer | number | boolean | null | enum | const
    properties: dict[str, "Node"] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    items: "Node | None" = None
    enum: list = field(default_factory=list)
    const: object = None
    min_items: int = 0
    max_items: int | None = None
    pattern_chars: frozenset | None = None  # restricted string alphabet


def compile_schema(schema: dict) -> Node:
    """Turn a JSON Schema subset into a Node tree.

    Supported: object (properties/required), array (items/minItems/maxItems),
    string, integer, number, boolean, null, enum, const. Unsupported keywords
    are rejected loudly rather than silently ignored, because a constraint that
    quietly does nothing is worse than one that errors.
    """
    if not isinstance(schema, dict):
        raise SchemaError(f"schema must be an object, got {type(schema).__name__}")

    if "const" in schema:
        return Node("const", const=schema["const"])
    if "enum" in schema:
        if not schema["enum"]:
            raise SchemaError("enum must have at least one value")
        return Node("enum", enum=list(schema["enum"]))

    t = schema.get("type")
    if t is None:
        raise SchemaError("schema needs a 'type', 'enum' or 'const'")
    if isinstance(t, list):
        raise SchemaError("union types are not supported; the automaton would "
                          "need to track several states at once")

    known = {"type", "properties", "required", "items", "minItems", "maxItems",
             "description", "title", "enum", "const"}
    unknown = set(schema) - known
    if unknown:
        raise SchemaError(f"unsupported schema keyword(s): {', '.join(sorted(unknown))}. "
                          f"silently ignoring them would produce output that "
                          f"validates here and fails downstream.")

    if t == "object":
        props = schema.get("properties") or {}
        if not props:
            raise SchemaError("object schemas need 'properties'; a free-form "
                              "object cannot constrain anything")
        required = schema.get("required", list(props))
        missing = [r for r in required if r not in props]
        if missing:
            raise SchemaError(f"required names not in properties: {missing}")
        # generation emits required keys in a fixed order, which is what keeps
        # the grammar regular. optional keys are not emitted at all.
        return Node("object",
                    properties={k: compile_schema(v) for k, v in props.items()},
                    required=list(required))

    if t == "array":
        if "items" not in schema:
            raise SchemaError("array schemas need 'items'")
        return Node("array", items=compile_schema(schema["items"]),
                    min_items=schema.get("minItems", 0),
                    max_items=schema.get("maxItems"))

    if t in ("string", "integer", "number", "boolean", "null"):
        return Node(t)

    raise SchemaError(f"unsupported type {t!r}")


# --------------------------------------------------------------------------
# The automaton itself.
#
# State is (a stack of frames). A frame is a node plus how far through it we
# are. `allowed()` returns the set of characters that may come next, and
# `advance()` consumes one character. That is the entire interface the token
# masker needs.
# --------------------------------------------------------------------------

@dataclass
class Frame:
    node: Node
    step: int = 0          # position within the node
    key_index: int = 0     # which required key we're on, for objects
    item_count: int = 0    # elements emitted, for arrays
    literal: str = ""      # remaining characters of a fixed literal
    seen_digits: int = 0
    seen_dot: bool = False
    seen_exp: bool = False
    first_digit: str = ""


class JsonAutomaton:
    """Character-level acceptor for one compiled schema.

    Deliberately not a validator you run afterwards — it has to answer
    "what's legal next" *before* each character exists, which is a different
    and stricter requirement.
    """

    def __init__(self, root: Node):
        self.root = root
        self.reset()

    def reset(self) -> None:
        self.stack: list[Frame] = [Frame(self.root)]
        self.emitted: list[str] = []
        self.done = False

    @property
    def text(self) -> str:
        return "".join(self.emitted)

    def clone(self) -> "JsonAutomaton":
        """Cheap copy, used to test-drive a candidate token without committing."""
        other = JsonAutomaton.__new__(JsonAutomaton)
        other.root = self.root
        other.stack = [Frame(f.node, f.step, f.key_index, f.item_count, f.literal,
                             f.seen_digits, f.seen_dot, f.seen_exp, f.first_digit)
                       for f in self.stack]
        other.emitted = list(self.emitted)
        other.done = self.done
        return other

    # ---------------------------------------------------------------- allowed

    def allowed(self) -> frozenset[str]:
        """Characters that may legally come next. Empty means the output is
        complete and nothing more may be emitted."""
        if self.done or not self.stack:
            return frozenset()
        return self._allowed_for(self.stack[-1])

    def _allowed_for(self, f: Frame) -> frozenset[str]:
        node = f.node

        # `literal` is always the prefix consumed so far, never the remainder.
        # Getting those two meanings mixed up wedges the automaton on the first
        # character of every fixed string, which is exactly what it did.
        if node.kind == "const":
            return self._next_chars([json.dumps(node.const)], f.literal)

        if node.kind == "enum":
            return self._next_chars([json.dumps(v) for v in node.enum], f.literal)

        if node.kind == "object":
            if f.step == 0:
                return frozenset("{")
            if f.step == 1:  # about to write a key, or close
                return frozenset('"' if f.key_index < len(node.required) else "}")
            if f.step == 2:
                return frozenset(":")
            if f.step == 4:  # after a value
                return frozenset("," if f.key_index < len(node.required) else "}")
            return frozenset()

        if node.kind == "array":
            if f.step == 0:
                return frozenset("[")
            if f.step == 1:
                out = set()
                if node.max_items is None or f.item_count < node.max_items:
                    out |= self._value_starts(node.items)
                if f.item_count >= node.min_items:
                    out.add("]")
                return frozenset(out)
            if f.step == 2:  # after an element
                out = set()
                if node.max_items is None or f.item_count < node.max_items:
                    out.add(",")
                if f.item_count >= node.min_items:
                    out.add("]")
                return frozenset(out)
            if f.step == 3:
                # straight after a comma an element is required. folding this
                # into step 1 lets the mask emit ["a",] which parses nowhere.
                return frozenset(self._value_starts(node.items))
            return frozenset()

        if node.kind == "string":
            if f.step == 0:
                return frozenset('"')
            if f.step == 1:
                return frozenset(STRING_CHARS | {'"', "\\"})
            if f.step == 2:  # just saw a backslash
                return frozenset(ESCAPES)
            if f.step in (3, 4, 5, 6):  # \uXXXX
                return frozenset(HEX)
            return frozenset()

        if node.kind in ("integer", "number"):
            out = set()
            if f.step == 0:
                out |= DIGITS | {"-"}
            else:
                # JSON forbids leading zeros: int = "0" / (digit1-9 *DIGIT).
                # allowing them produces text that this automaton accepts and
                # json.loads rejects.
                leading_zero = f.seen_digits == 1 and f.first_digit == "0"
                if not leading_zero:
                    out |= DIGITS
                if f.seen_digits:
                    if node.kind == "number" and not f.seen_dot and not f.seen_exp:
                        out.add(".")
                    # exponents belong to "number", not "integer": 1e5 is a
                    # float to json.loads, so allowing it under an integer
                    # schema produces something the mask accepts and a
                    # validator rejects.
                    if node.kind == "number" and not f.seen_exp:
                        out |= {"e", "E"}
                    # and the exponent is capped, because 9e999999 parses to
                    # inf, which is valid JSON syntax and not a number
                    if f.seen_exp and f.seen_digits >= MAX_EXPONENT_DIGITS:
                        out -= DIGITS
                    # a complete number may end here, signalled by the parent
                    out |= self._terminators()
            if f.step == 1 and not f.seen_digits:  # just after '-'
                out = set(DIGITS)
            return frozenset(out)

        if node.kind == "boolean":
            return self._next_chars(["true", "false"], f.literal)

        if node.kind == "null":
            return self._next_chars(["null"], f.literal)

        return frozenset()

    @staticmethod
    def _next_chars(options: list[str], consumed: str) -> frozenset[str]:
        """Characters that may follow `consumed` in any of the options.

        More than one option can still be live at once ("t"/"f" for a boolean,
        several enum members sharing a prefix), which is why this returns a set
        rather than a single character.
        """
        return frozenset(
            opt[len(consumed)] for opt in options
            if len(opt) > len(consumed) and opt.startswith(consumed)
        )

    @staticmethod
    def _value_starts(node: Node) -> set[str]:
        if node.kind == "object":
            return {"{"}
        if node.kind == "array":
            return {"["}
        if node.kind == "string":
            return {'"'}
        if node.kind in ("integer", "number"):
            return set(DIGITS) | {"-"}
        if node.kind == "boolean":
            return {"t", "f"}
        if node.kind == "null":
            return {"n"}
        if node.kind == "const":
            return {json.dumps(node.const)[0]}
        if node.kind == "enum":
            return {json.dumps(v)[0] for v in node.enum}
        return set()

    def _terminators(self) -> set[str]:
        """What may follow a finished scalar, which depends on the parent.

        The object case has to account for keys still required. Returning
        {",", "}"} unconditionally lets the mask close an object early, and the
        result parses as JSON while silently missing required fields — a mask
        that is too permissive fails in exactly this quiet way.
        """
        if len(self.stack) < 2:
            return set()
        parent = self.stack[-2]
        if parent.node.kind == "object":
            # popping this scalar will advance key_index by one
            remaining = len(parent.node.required) - (parent.key_index + 1)
            return {","} if remaining > 0 else {"}"}
        if parent.node.kind == "array":
            out = set()
            if parent.node.max_items is None or parent.item_count + 1 < parent.node.max_items:
                out.add(",")
            if parent.item_count + 1 >= parent.node.min_items:
                out.add("]")
            return out
        return set()

    def depth(self) -> int:
        return len(self.stack)

    # ---------------------------------------------------------------- advance

    def accepts(self, ch: str) -> bool:
        return ch in self.allowed()

    def advance(self, ch: str) -> bool:
        """Consume one character. Returns False if it wasn't legal."""
        if ch not in self.allowed():
            return False
        self.emitted.append(ch)
        self._step(ch)
        if not self.stack:
            self.done = True
        return True

    def feed(self, text: str) -> bool:
        for ch in text:
            if not self.advance(ch):
                return False
        return True

    def _step(self, ch: str) -> None:
        f = self.stack[-1]
        node = f.node

        if node.kind in ("const", "enum"):
            f.literal += ch
            options = ([json.dumps(node.const)] if node.kind == "const"
                       else [json.dumps(v) for v in node.enum])
            if f.literal in options:
                self._pop()
            return

        if node.kind == "object":
            if f.step == 0 and ch == "{":
                f.step = 1
            elif f.step == 1 and ch == "}":
                self._pop()
            elif f.step == 1 and ch == '"':
                # keys are fixed, so push a literal frame for the exact name
                key = node.required[f.key_index]
                f.step = 2
                self.stack.append(Frame(Node("const", const=key), literal='"'))
            elif f.step == 2 and ch == ":":
                f.step = 3
                self.stack.append(Frame(node.properties[node.required[f.key_index]]))
            elif f.step == 4 and ch == ",":
                f.step = 1
            elif f.step == 4 and ch == "}":
                self._pop()
            return

        if node.kind == "array":
            if f.step == 0 and ch == "[":
                f.step = 1
            elif f.step == 1 and ch == "]":
                self._pop()
            elif f.step == 1:
                f.step = 2
                child = Frame(node.items)
                self.stack.append(child)
                child_started = self._start_value(child, ch)
                if child_started:
                    return
            elif f.step == 2 and ch == ",":
                f.step = 3
            elif f.step == 2 and ch == "]":
                self._pop()
            elif f.step == 3:
                f.step = 2
                self.stack.append(Frame(node.items))
                self._step(ch)
            return

        if node.kind == "string":
            if f.step == 0 and ch == '"':
                f.step = 1
            elif f.step == 1 and ch == '"':
                self._pop()
            elif f.step == 1 and ch == "\\":
                f.step = 2
            elif f.step == 2:
                f.step = 3 if ch == "u" else 1
            elif f.step in (3, 4, 5):
                f.step += 1
            elif f.step == 6:
                f.step = 1
            return

        if node.kind in ("integer", "number"):
            if ch in DIGITS:
                f.step = 1
                if f.seen_digits == 0:
                    f.first_digit = ch
                f.seen_digits += 1
            elif ch == "-":
                f.step = 1
            elif ch == ".":
                f.seen_dot = True
                f.seen_digits = 0
                f.first_digit = ""
            elif ch in "eE":
                f.seen_exp = True
                f.seen_digits = 0
                f.first_digit = ""
            else:
                # a terminator: the number is finished, hand the char to the parent
                self._pop()
                if self.stack:
                    self._step(ch)
            return

        if node.kind in ("boolean", "null"):
            f.literal += ch
            if f.literal in ("true", "false", "null"):
                self._pop()
            return

    def _start_value(self, child: Frame, ch: str) -> bool:
        return self.advance_into(child, ch)

    def advance_into(self, child: Frame, ch: str) -> bool:
        self._step(ch)
        return True

    def _pop(self) -> None:
        self.stack.pop()
        if not self.stack:
            self.done = True
            return
        parent = self.stack[-1]
        if parent.node.kind == "object":
            if parent.step == 2:
                pass  # just finished a key, waiting for ':'
            elif parent.step == 3:
                parent.step = 4
                parent.key_index += 1
        elif parent.node.kind == "array":
            parent.item_count += 1

    # ---------------------------------------------------------------- helpers

    def is_complete(self) -> bool:
        return self.done

    def state_key(self) -> tuple:
        """A hashable digest of the automaton state.

        Two positions with the same key allow the same characters, which is
        what makes the mask cache in mask.py work.
        """
        # item_count may only be clamped when the array is unbounded. with a
        # finite maxItems, "2 elements" and "3 elements" differ in whether a
        # comma is still legal, and collapsing them makes the cache hand back a
        # mask that permits one element too many.
        return tuple(
            (id(f.node), f.step, f.key_index,
             (min(f.item_count, 2) if f.node.kind == "array" and f.node.max_items is None
              else f.item_count),
             f.literal, min(f.seen_digits, DIGIT_STATE_CAP), f.seen_dot,
             f.seen_exp, f.first_digit)
            for f in self.stack
        )


def iter_schema_nodes(node: Node) -> Iterator[Node]:
    yield node
    for child in node.properties.values():
        yield from iter_schema_nodes(child)
    if node.items is not None:
        yield from iter_schema_nodes(node.items)
