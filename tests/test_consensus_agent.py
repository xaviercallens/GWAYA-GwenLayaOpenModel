"""ConsensusRepairAgent with a scripted generator and the real bwrap sandbox."""
import os
os.environ["GWAYA_ALLOW_UNISOLATED"] = "1"

import pytest

from gwaya.consensus_agent import ConsensusRepairAgent
PUBLIC = "assert f(2) == 4\n"
GOAL = "Write f(x) that doubles x."


class Script:
    """Routes by prompt content: self-test request, repair request, or first-round sampling."""

    def __init__(self, first, repair=None, selftests=""):
        self.first, self.repair, self.selftests = list(first), list(repair or []), selftests
        self.prompts: list[str] = []

    def __call__(self, prompt, temperature=0.2, max_tokens=512, **_):
        self.prompts.append(prompt)
        if "additional assert" in prompt:
            return f"```python\n{self.selftests}\n```"
        pool = self.repair if "--- REPAIR" in prompt else self.first
        code = pool.pop(0) if len(pool) > 1 else pool[0]
        return f"```python\n{code}\n```"


def agent(gen, **kw):
    opts = {"n_candidates": 3, "repair_candidates": 2, "max_repair_rounds": 2, **kw}
    return ConsensusRepairAgent(gen, **opts)


def test_consensus_rejects_public_passing_wrong_candidate():
    wrong, right1, right2 = "def f(x):\n    return x + 2\n", "def f(x):\n    return x * 2\n", "def f(x):\n    return x + x\n"
    gen = Script([wrong, right1, right2], selftests="assert f(3) == 6\nassert f(5) == 10")
    ans = agent(gen).solve(GOAL, PUBLIC)
    assert ans.verified and ans.level == "PUBLIC_TEST_AND_CONSENSUS"
    assert ans.code.strip() in (right1.strip(), right2.strip())  # the wrong one passes the public test but not the consensus


def test_without_selftests_the_wrong_candidate_is_selected():
    wrong, right = "def f(x):\n    return x + 2\n", "def f(x):\n    return x * 2\n"
    ans = agent(Script([wrong, right]), use_selftests=False).solve(GOAL, PUBLIC)
    assert ans.code.strip() == wrong.strip()  # documents the failure mode H1 fixes


def test_hallucinated_candidate_is_rejected_and_repaired():
    bad = "import numpyx\ndef f(x):\n    return numpyx.double(x)\n"
    good = "def f(x):\n    return x * 2\n"
    gen = Script([bad], repair=[good], selftests="")
    ans = agent(gen).solve(GOAL, PUBLIC)
    assert ans.verified and ans.rounds >= 1 and "numpyx" not in ans.code
    assert any("unresolvable import" in f for f in ans.flags_seen)


def test_grounding_off_lets_the_hallucination_reach_the_sandbox_instead():
    bad = "import math\ndef f(x):\n    return math.double(x)\n"
    ans = agent(Script([bad]), grounding=False, use_selftests=False, max_repair_rounds=0).solve(GOAL, PUBLIC)
    assert not ans.verified  # still never verified, but not rejected statically
    assert ans.flags_seen == []


def test_abstains_when_nothing_passes():
    wrong = "def f(x):\n    return x + 1\n"
    ans = agent(Script([wrong], repair=[wrong])).solve(GOAL, PUBLIC)
    assert ans.verified is False and ans.status == "UNVERIFIED" and ans.level == "UNVERIFIED"
    assert ans.code  # best attempt is still returned, clearly marked unverified


def test_full_context_repair_shows_whole_code_and_actual_value():
    wrong = "def f(x):\n    # " + "padding " * 40 + "\n    return x + 1\n"
    right = "def f(x):\n    return x * 2\n"
    gen = Script([wrong], repair=[right])
    ans = agent(gen, use_selftests=False).solve(GOAL, PUBLIC)
    repair_prompt = next(p for p in gen.prompts if "--- REPAIR" in p)
    assert "return x + 1" in repair_prompt and "f(2) returned 3" in repair_prompt
    assert ans.verified


def test_old_style_repair_truncates_code():
    wrong = "def f(x):\n    # " + "padding " * 40 + "\n    return x + 1\n"
    gen = Script([wrong], repair=["def f(x):\n    return x * 2\n"])
    agent(gen, use_selftests=False, full_context_repair=False).solve(GOAL, PUBLIC)
    repair_prompt = next(p for p in gen.prompts if "REPAIR FEEDBACK" in p)
    assert "return x + 1" not in repair_prompt and "[truncated]" in repair_prompt


def test_no_public_test_gives_consensus_only_level():
    r1, r2 = "def f(x):\n    return x * 2\n", "def f(x):\n    return x + x\n"
    ans = agent(Script([r1, r2], selftests="assert f(3) == 6\nassert f(4) == 8")).solve(GOAL, None)
    assert ans.verified and ans.level == "CONSENSUS_ONLY"
