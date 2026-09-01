"""Intent is read from the spec and inferred from nothing.

``contract/spec.py`` opens by claiming "everything downstream reads this object
and nothing else for intent". That is the foundation the gate stands on: a
guide's verdict turns on whether a hit is REQUIRED, DESIRED or FORBIDDEN, and if
any stage could *infer* intent -- from a locus name, a biotype, a sequence
resemblance -- then two stages could disagree about what a hit is, and the
disagreement would surface as a confident verdict rather than as an error.

The claim holds today, and the sweep that checked the other uniqueness claims in
this codebase found two of them false, so it is worth holding still rather than
re-deriving. The shape of the invariant:

* a ``SearchHit`` receives ``target_class`` in exactly one place,
  ``c_search.annotate_hit``;
* that place reads it from ``spec.contract``, never from the hit, the locus, or
  the sequence;
* an unannotated hit is *not* intended, which is the safe default -- an
  off-target mistaken for a target inflates coverage, the direction this
  package exists to prevent.
"""

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cas13guide"


def assignments_of(field: str) -> dict[str, list[int]]:
    """Every ``field=`` keyword argument in the package, by module."""
    found: dict[str, list[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == field
        ]
        if lines:
            found[path.relative_to(SRC).as_posix()] = lines
    return found


class TestOnlyOnePlaceAssignsIntentToAHit(unittest.TestCase):
    """``TargetSpec(target_class=...)`` builds the contract; that is different
    from deciding what a *hit* is, which is what this guards."""

    #: Modules that construct contract objects. Declaring intent when you build
    #: the spec is the point of the spec.
    CONTRACT_BUILDERS = {
        "contract/presets.py", "services/fasta.py", "services/workflow.py", "demo.py",
    }

    def test_annotate_hit_is_the_only_stage_that_labels_a_hit(self):
        sites = assignments_of("target_class")
        stages = sorted(set(sites) - self.CONTRACT_BUILDERS)
        self.assertEqual(stages, ["pipeline/c_search.py"], sites)

    def test_and_it_takes_the_label_from_the_contract(self):
        source = (SRC / "pipeline" / "c_search.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "annotate_hit"
        )
        body = ast.unparse(fn)
        self.assertIn("spec.contract.get", body)
        self.assertIn("target.target_class", body)

    def test_the_contract_builders_are_still_where_we_think(self):
        """Stops the exclusion list above from quietly hiding a new stage."""
        sites = set(assignments_of("target_class"))
        self.assertTrue(self.CONTRACT_BUILDERS <= sites | {"demo.py"}, sites)


class TestAnUnlabelledHitIsNotIntended(unittest.TestCase):
    """The default has to fall the safe way.

    An off-target mistaken for a target inflates coverage; a target mistaken for
    an off-target inflates hazard. Only the first is a claim about efficacy the
    evidence does not support, so ``None`` must mean "not intended".
    """

    def _evidence(self, target_class):
        import dataclasses
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from helpers import make_hit  # noqa: F401

        from cas13guide.metrics.bounds import Interval
        from cas13guide.models.evidence import PairEvidence

        hit = make_hit("g1", "L1")
        hit = dataclasses.replace(hit, target_class=target_class)
        return PairEvidence(
            guide_id="g1", locus_id="L1", context="reference", hit=hit,
            activity=Interval(0.9, 0.9, 0.95),
            expression=Interval(0.5, 0.5, 0.5), hazard_weight=0.0,
        )

    def test_an_unannotated_hit_is_an_off_target(self):
        self.assertFalse(self._evidence(None).is_intended)

    def test_a_required_hit_is_intended(self):
        from cas13guide.contract.enums import TargetClass

        self.assertTrue(self._evidence(TargetClass.REQUIRED).is_intended)

    def test_a_forbidden_hit_is_not_intended(self):
        from cas13guide.contract.enums import TargetClass

        self.assertFalse(self._evidence(TargetClass.FORBIDDEN).is_intended)


if __name__ == "__main__":
    unittest.main()
