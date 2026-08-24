"""Entry point for ``python -m cas13guide``.

The ``cas13guide`` console script is the documented way in, but it only works
once its directory is on PATH -- which a ``pip install --user`` on Windows
routinely fails to arrange. ``python -m`` needs nothing but the interpreter that
installed the package, so it is the invocation that always works.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
