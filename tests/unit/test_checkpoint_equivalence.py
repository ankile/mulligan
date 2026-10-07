"""Golden regression for released sim agents.

One released checkpoint per kind (idql-agent, divl-agent) is downloaded anonymously
from HF and loaded through the release loader. The probe outputs (critic/value heads, best-of-N
samples and the executed action chunk) must equal, bitwise, the recorded references (CPU,
deterministic algorithms, the same torch version).
"""

from __future__ import annotations

import importlib.util
import json
import pickletools
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "checkpoint_equivalence"
CASES = json.loads((FIXTURES / "cases.json").read_text())
REFERENCES = json.loads((FIXTURES / "references.json").read_text())


def _probe():
    spec = importlib.util.spec_from_file_location("_ckpt_probe", FIXTURES / "probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pickled_globals(policy_pt: Path) -> set[str]:
    with zipfile.ZipFile(policy_pt) as zf:
        data = zf.read(next(n for n in zf.namelist() if n.endswith("data.pkl")))
    names, strings = set(), []
    for opcode, arg, _ in pickletools.genops(data):
        if isinstance(arg, str):
            strings.append(arg)
        if opcode.name == "STACK_GLOBAL":
            names.add(".".join(strings[-2:]))
        elif opcode.name == "GLOBAL":
            names.add(arg.replace(" ", "."))
    return names


@pytest.mark.network
@pytest.mark.parametrize("kind", sorted(CASES))
def test_released_checkpoint_matches_source_outputs(kind):
    from mulligan import apply_runtime_patches
    from mulligan.release.hub import resolve_checkpoint
    from mulligan.utils.load_pretrained import load_policy_from_checkpoint

    case = CASES[kind]
    reference = REFERENCES["cases"][kind]
    # The local-version suffix (+cu128, +cpu) names the build, not the CPU kernels.
    if _release(torch.__version__) != _release(REFERENCES["recorded_with"]["torch"]):
        pytest.fail(
            f"torch {torch.__version__} differs from the reference torch "
            f"{REFERENCES['recorded_with']['torch']}: switch this test to atol=1e-6, rtol=1e-5 "
            "and record the change with the references"
        )

    # The pin comes from release/revisions.json; the weights must be the recorded ones.
    ckpt_dir = resolve_checkpoint(f"hf://{case['repo']}/{case['subfolder']}")
    assert _sha256(ckpt_dir / "policy.pt") == reference["policy_sha256"]
    # The released pickle names this package's config classes (no source-repository names).
    names = _pickled_globals(ckpt_dir / "policy.pt")
    assert any(name.startswith("mulligan.configs.policy.") for name in names)

    probe = _probe()
    probe.configure_determinism()
    apply_runtime_patches()
    policy, _, _ = load_policy_from_checkpoint(ckpt_dir, device="cpu")
    assert type(policy.config).__name__ == reference["config_class"]
    assert type(policy.config).__module__ == "mulligan.configs.policy"

    got = probe.probe(policy)
    want = torch.load(FIXTURES / f"{kind}.pt", map_location="cpu", weights_only=True)
    assert sorted(got) == sorted(want) == reference["keys"]
    for key in want:
        assert got[key].dtype == want[key].dtype and got[key].shape == want[key].shape, key
        assert torch.equal(got[key], want[key]), f"{kind}: {key} differs from the source output"


def _release(version: str) -> str:
    return version.split("+", 1)[0]


def test_torch_version_check_ignores_the_build_suffix():
    assert _release("2.11.0+cu128") == _release("2.11.0+cpu") == _release("2.11.0") == "2.11.0"
    assert _release("2.11.1") != _release("2.11.0+cu128")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_reference_fixtures_are_plain_tensors():
    """The fixtures load with weights_only=True (no pickled code)."""
    for kind in CASES:
        payload = torch.load(FIXTURES / f"{kind}.pt", map_location="cpu", weights_only=True)
        assert payload and all(isinstance(v, torch.Tensor) for v in payload.values())
        assert sorted(payload) == REFERENCES["cases"][kind]["keys"]
