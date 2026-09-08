"""A threshold that cannot fire has to say so.

The gate skips ``TARGET_INACCESSIBLE`` when accessibility is ``None``, and
skips ``DUPLEX_ENERGY_OUT_OF_RANGE`` when the duplex energy is ``None``. Both
are right: absent evidence is not bad evidence, and flagging every guide
because nobody folded anything would be worse than saying nothing.

The consequence, though, was that ``--min-accessibility 0.9`` without ViennaRNA
installed asked for a gate that could not fire, and nothing said so. A generic
"no structural accessibility was computed" warning was emitted either way --
identical whether the user had asked for a floor or never mentioned one -- so
the one reader who needed to know got the same sentence as the one who did not.

Found while testing the three gate paths that 950 tests never produced: the
flag is plumbed correctly end to end, and the silence was the defect.
"""

import random
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _limitations(*, min_accessibility=0.0, structure=False):
    from cas13guide.services.workflow import DesignRequest, run_request

    rng = random.Random(7)
    sequence = "".join(rng.choice("ACGU") for _ in range(240))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.fa"
        path.write_text(f">t1\n{sequence}\n", encoding="utf-8")
        outcome = run_request(
            DesignRequest(fasta_path=str(path), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d",
                          use_structure=structure, use_signal=False,
                          min_accessibility=min_accessibility)
        )
    return {limitation.id: limitation for limitation in outcome.limitations}


class TestAnUnenforceableAccessibilityFloor(unittest.TestCase):
    def test_it_is_critical_and_names_the_number_that_went_unused(self):
        from cas13guide.services.limitations import Severity

        found = _limitations(min_accessibility=0.99, structure=False)
        self.assertIn("accessibility-floor-inert", found)

        limitation = found["accessibility-floor-inert"]
        self.assertIs(limitation.severity, Severity.CRITICAL)
        self.assertIn("0.99", limitation.text)
        self.assertIn("NOT", limitation.text)

    def test_the_generic_warning_gives_way_to_it(self):
        """Two limitations about one absence is noise. The specific one wins,
        because it says everything the generic one did and more."""
        found = _limitations(min_accessibility=0.99, structure=False)
        self.assertNotIn("no-accessibility", found)

    def test_a_run_that_asked_for_no_floor_still_gets_the_plain_warning(self):
        from cas13guide.services.limitations import Severity

        found = _limitations(min_accessibility=0.0, structure=False)
        self.assertIn("no-accessibility", found)
        self.assertIs(found["no-accessibility"].severity, Severity.WARNING)
        self.assertNotIn("accessibility-floor-inert", found)

    def test_nothing_is_reported_when_the_floor_can_actually_fire(self):
        """The point is an *unenforceable* threshold. With a folding engine the
        floor is enforced, and warning about it would be false."""
        found = _limitations(min_accessibility=0.99, structure=True)
        self.assertNotIn("accessibility-floor-inert", found)
        self.assertNotIn("no-accessibility", found)


class TestAnUnenforceableDuplexWindow(unittest.TestCase):
    """The same defect one field over, reached through the API rather than a
    flag. Tested at the limitations layer because no CLI option sets it."""

    def _found(self, **thresholds):
        from cas13guide.services.limitations import limitations_for_run
        from cas13guide.services.registry import (
            HeuristicCas13Model,
            ModelRegistry,
            NullAccessibilityModel,
        )

        spec = make_spec(**thresholds)
        registry = ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel())
        return {l.id: l for l in limitations_for_run(spec, registry=registry)}

    def test_a_window_with_no_energy_model_is_reported_as_not_applied(self):
        from cas13guide.services.limitations import Severity

        found = self._found(duplex_dg_per_nt_min=-0.5, duplex_dg_per_nt_max=-0.2)
        self.assertIn("duplex-window-inert", found)
        self.assertIs(found["duplex-window-inert"].severity, Severity.CRITICAL)
        self.assertIn("NOT", found["duplex-window-inert"].text)

    def test_a_one_sided_window_counts_too(self):
        """Both bounds are independently optional, so either one alone is a
        window the user expects to be enforced."""
        for thresholds in ({"duplex_dg_per_nt_min": -0.5},
                           {"duplex_dg_per_nt_max": -0.2}):
            with self.subTest(**thresholds):
                self.assertIn("duplex-window-inert", self._found(**thresholds))

    def test_no_window_means_nothing_to_report(self):
        self.assertNotIn("duplex-window-inert", self._found())


if __name__ == "__main__":
    unittest.main()
