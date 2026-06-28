import numpy as np

from robot_eyes.behavior import EMOTES, AdaptivePolicy, Emote
from robot_eyes.config import Mood


def test_emote_library_has_expected_actions():
    names = {e.name for e in EMOTES}
    assert {"mirror", "smile", "wink_left", "wink_right"} <= names
    actions = {e.action for e in EMOTES}
    assert "wink_left" in actions and "blink" in actions and "look" in actions


def test_exploit_with_empty_table_picks_mirror():
    pol = AdaptivePolicy(EMOTES, epsilon=0.0, seed=0)
    emote, explored = pol.select("anon", "neutral")
    assert emote.name == "mirror"
    assert explored is False


def test_epsilon_one_always_explores():
    pol = AdaptivePolicy(EMOTES, epsilon=1.0, seed=0)
    assert all(pol.select("anon", "neutral")[1] for _ in range(20))


def test_update_is_incremental_mean():
    pol = AdaptivePolicy(EMOTES, epsilon=0.0, seed=0)
    e = pol.by_name["smile"]
    pol.update("u", "Sad", e, 1.0)
    pol.update("u", "Sad", e, 0.0)
    val, n = pol.value("u", "Sad", "smile")
    assert n == 2
    assert np.isclose(val, 0.5)


def test_policy_converges_to_high_reward_emote():
    pol = AdaptivePolicy(EMOTES, epsilon=0.2, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(400):
        e, _ = pol.select("u", "Sad")
        r = 0.9 if e.name == "smile" else 0.1
        pol.update("u", "Sad", e, r + rng.normal(0, 0.02))
    best = max(EMOTES, key=lambda e: pol.value("u", "Sad", e.name)[0])
    assert best.name == "smile"
    # exploitation now favours smile.
    picks = [pol.select("u", "Sad")[0].name for _ in range(200)]
    assert picks.count("smile") > 120


def test_context_separation():
    pol = AdaptivePolicy(EMOTES, epsilon=0.0, seed=0)
    pol.update("u", "Happy", pol.by_name["smile"], 1.0)
    # 'smile' is great when Happy, but unknown (0) for a different context.
    assert pol.value("u", "Happy", "smile")[0] == 1.0
    assert pol.value("u", "Sad", "smile") == (0.0, 0)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "policy.txt"
    pol = AdaptivePolicy(EMOTES, epsilon=0.2, seed=0, path=path)
    pol.update("userA", "Sad", pol.by_name["wink_left"], 0.7)
    pol.update("userA", "Sad", pol.by_name["smile"], 0.9)
    pol.save()

    pol2 = AdaptivePolicy(EMOTES, path=path)
    assert pol2.value("userA", "Sad", "wink_left")[0] == 0.7
    assert pol2.value("userA", "Sad", "smile")[0] == 0.9


def test_custom_emote_dataclass_defaults():
    e = Emote("x", Mood.HAPPY)
    assert e.action is None and e.look == (0.0, 0.0) and e.duration == 2.5
