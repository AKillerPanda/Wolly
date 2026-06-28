"""
behavior.py - Adaptive eye behaviour: pick expressive "emotes" and learn which
ones get a better reaction from the person in front of the camera.

Instead of mirroring the user's emotion 1:1, the eyes choose from a small library
of emotes (smile, wink, playful look-around, surprised blink, sympathy, calm).
The choice is an epsilon-greedy contextual bandit:

  * ~(1 - epsilon) of the time it EXPLOITS: play the best-known emote for the
    current context.
  * ~epsilon of the time (default 0.2) it EXPLORES: try a different emote, so the
    display keeps changing and can discover something that works better.

After an emote has played for its duration, the caller reports a scalar reward
(the user's happiness during it). The bandit updates that emote's running value
for this (identity, context) pair, so over time it favours emotes that make this
particular person happier. The learned table persists to a readable text file.

The policy is deliberately decoupled from the emotion/vision code: it just needs
an identity string, a context string, and a reward number.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Mood


@dataclass(frozen=True)
class Emote:
    """One expressive action the eyes can take.

    mood   : the eye mood to hold, or None meaning "mirror the user's mood".
    action : one-shot motion at the start: None / 'blink' / 'wink_left' /
             'wink_right' / 'look'.
    look   : (dx, dy) gaze target for the 'look' action, each in [-1, 1].
    """
    name: str
    mood: Mood | None
    action: str | None = None
    look: tuple[float, float] = (0.0, 0.0)
    duration: float = 2.5


# The action set. Most aim to lift mood; "mirror" and "sympathy" stay empathic so
# the eyes are not relentlessly grinning.
EMOTES: list[Emote] = [
    Emote("mirror", None, None),
    Emote("smile", Mood.HAPPY, None),
    Emote("wink_left", Mood.HAPPY, "wink_left"),
    Emote("wink_right", Mood.HAPPY, "wink_right"),
    Emote("playful_look", Mood.NEUTRAL, "look", look=(0.9, -0.2)),
    Emote("surprise", Mood.SURPRISED, "blink", duration=1.8),
    Emote("sympathy", Mood.SAD, None),
    Emote("calm", Mood.NEUTRAL, "blink", duration=3.0),
]


class AdaptivePolicy:
    """Epsilon-greedy contextual bandit over a list of emotes, with txt persistence."""

    def __init__(self, emotes: list[Emote] = EMOTES, epsilon: float = 0.2,
                 seed: int | None = None, path: str | Path | None = None):
        self.emotes = emotes
        self.by_name = {e.name: e for e in emotes}
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.path = Path(path) if path else None
        # Q[(identity, context, emote_name)] = [value, count]
        self.Q: dict[tuple[str, str, str], list[float]] = {}
        self._dirty = 0
        if self.path and self.path.exists():
            self.load()

    # ----- selection -----

    def select(self, identity: str, context: str) -> tuple[Emote, bool]:
        """Choose an emote for (identity, context). Returns (emote, explored?)."""
        if self.rng.random() < self.epsilon:
            return self.emotes[int(self.rng.integers(len(self.emotes)))], True

        # Exploit: highest value; unseen arms count as 0. Tie-break -> 'mirror',
        # then a random pick among the top so behaviour still varies.
        best_val = -np.inf
        scored: list[tuple[float, Emote]] = []
        for e in self.emotes:
            v = self.Q.get((identity, context, e.name), [0.0, 0])[0]
            scored.append((v, e))
            best_val = max(best_val, v)
        top = [e for v, e in scored if v >= best_val - 1e-9]
        if any(e.name == "mirror" for e in top):
            return self.by_name["mirror"], False
        return top[int(self.rng.integers(len(top)))], False

    # ----- learning -----

    def update(self, identity: str, context: str, emote: Emote, reward: float) -> None:
        key = (identity, context, emote.name)
        val, n = self.Q.get(key, [0.0, 0])
        n += 1
        val += (reward - val) / n          # incremental sample mean
        self.Q[key] = [val, n]
        self._dirty += 1
        if self.path and self._dirty >= 5:
            self.save()

    def value(self, identity: str, context: str, emote_name: str) -> tuple[float, int]:
        v, n = self.Q.get((identity, context, emote_name), [0.0, 0])
        return float(v), int(n)

    # ----- persistence (readable txt: identity | context | emote | value | count) -----

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# affect-pi adaptive emote policy. Higher value = better user reaction.",
            "# format: identity | context | emote | value | count",
        ]
        for (ident, ctx, name), (val, n) in sorted(self.Q.items()):
            lines.append(f"{ident} | {ctx} | {name} | {val:.5f} | {int(n)}")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._dirty = 0

    def load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 5:
                continue
            ident, ctx, name, val, n = parts
            self.Q[(ident, ctx, name)] = [float(val), int(n)]
