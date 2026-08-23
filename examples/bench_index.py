"""Re-measure the index cost-model constants on this machine.

The constants in ``services/index_selection.py`` were fitted on one machine.
Run this where the pipeline will actually live and update them if the numbers
have moved -- the model's *shape* holds (scan linear in corpus, FM query flat,
build linear), only the coefficients are hardware-specific.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cas13guide.contract.enums import Universe
from cas13guide.services.fmindex import FMIndexSearchIndex
from cas13guide.services.indexes import ExhaustiveScanIndex
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.util.seq import revcomp

UNIVERSES = frozenset({Universe.CODING})


def corpus(total_bases: int, per_locus: int = 2000, seed: int = 5) -> Transcriptome:
    rng = random.Random(seed)
    t = Transcriptome("bench", "v1")
    for i in range(total_bases // per_locus):
        t.add(
            Locus(
                f"L{i}",
                "".join(rng.choice("AUGC") for _ in range(per_locus)),
                Universe.CODING,
                gene=f"G{i}",
                expression=10.0,
            )
        )
    return t


def main(sizes=(600_000, 1_500_000, 3_000_000), n_queries: int = 5, k: int = 4) -> None:
    rng = random.Random(11)
    print(f"{'corpus':>12} {'build':>9} {'FM/query':>10} {'scan/query':>11} "
          f"{'build s/Mb':>11} {'scan s/Mb/q':>12}")
    for total in sizes:
        world = corpus(total)
        spacers = []
        for _ in range(n_queries):
            locus = rng.choice(list(world.loci.values()))
            start = rng.randrange(0, len(locus.sequence) - 22)
            spacers.append(revcomp(locus.sequence[start : start + 22]))

        t0 = time.perf_counter()
        fm = FMIndexSearchIndex(world, UNIVERSES)
        build = time.perf_counter() - t0
        scan = ExhaustiveScanIndex(world, UNIVERSES)

        t0 = time.perf_counter()
        fm_hits = [fm.search("g", s, k) for s in spacers]
        fm_q = (time.perf_counter() - t0) / len(spacers)

        t0 = time.perf_counter()
        scan_hits = [scan.search("g", s, k) for s in spacers]
        scan_q = (time.perf_counter() - t0) / len(spacers)

        def sig(r):
            return {(h.locus_id, h.start, h.duplex.n_mismatch) for h in r.hits}

        assert all(sig(a) == sig(b) for a, b in zip(fm_hits, scan_hits)), "indexes disagree"

        mb = total / 1_000_000
        print(f"{total:>12,} {build:>8.2f}s {fm_q:>9.3f}s {scan_q:>10.3f}s "
              f"{build / mb:>11.2f} {scan_q / mb:>12.3f}")

    print("\nUpdate BUILD_SECONDS_PER_MB / SCAN_SECONDS_PER_MB_PER_QUERY / "
          "FM_SECONDS_PER_QUERY in services/index_selection.py from these.")


if __name__ == "__main__":
    main()
