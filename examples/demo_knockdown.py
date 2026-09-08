"""End-to-end demo: knock down a transcript in cell culture.

The demo itself lives in :mod:`cas13guide.demo` so that ``cas13guide demo``
works for anyone who installed the package, and not only inside a source
checkout: ``examples/`` does not ship in the wheel, and the subcommand used to
reach for this file by walking two directories up from ``cli.py``.

This file stays because the README documents running it directly, and because a
source checkout should be able to try the thing without installing first --
hence the ``sys.path`` line, which is this script's business rather than the
package's.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cas13guide.demo import build_world, main, seed_calibration  # noqa: E402,F401

if __name__ == "__main__":
    main()
