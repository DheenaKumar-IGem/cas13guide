"""What an installed copy can do, as opposed to a source checkout.

Every earlier pass ran the product from the repository, where ``examples/`` is
one directory up and ``src/`` is importable. Nobody had built a wheel, installed
it into a clean interpreter, and run the documented commands from somewhere
else -- and that is a different program: the wheel ships ``src/cas13guide/`` and
nothing beside it.

It found a subcommand advertised in ``--help`` that died with a traceback for
every user who had installed the package rather than cloned it.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestTheDemoShipsWithThePackage(unittest.TestCase):
    """``cas13guide demo`` reached outside the distribution for its own code.

    ``_demo`` put ``<cli.py>/../../examples`` on ``sys.path`` and imported
    ``demo_knockdown``. In a checkout that resolves; from site-packages it
    resolves to ``Lib/examples``, which does not exist, because ``examples/`` is
    not part of the wheel. So the subcommand named in ``--help`` and in the
    module docstring raised ``ModuleNotFoundError`` and exited 1 for anyone who
    pip-installed the package.
    """

    def test_the_demo_is_inside_the_package(self):
        self.assertTrue((REPO / "src" / "cas13guide" / "demo.py").is_file())

    def test_it_imports_without_any_path_manipulation(self):
        """The property that was broken: importable as a package module."""
        result = subprocess.run(
            [sys.executable, "-c", "import cas13guide.demo; print(cas13guide.demo.main)"],
            capture_output=True, text=True, timeout=300, cwd=str(REPO),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("function main", result.stdout)

    def test_the_cli_no_longer_reaches_for_the_examples_directory(self):
        source = (REPO / "src" / "cas13guide" / "cli.py").read_text(encoding="utf-8")
        self.assertNotIn("import demo_knockdown", source)
        self.assertIn("from .demo import main", source)

    def test_the_package_never_puts_anything_on_sys_path(self):
        """A library adjusting sys.path is how this defect was possible.

        Scripts under ``examples/`` may do it -- that is their business. Nothing
        under ``src/cas13guide/`` should.
        """
        offenders = [
            path.relative_to(REPO).as_posix()
            for path in (REPO / "src" / "cas13guide").rglob("*.py")
            if "sys.path.insert" in path.read_text(encoding="utf-8")
            or "sys.path.append" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_the_example_script_still_runs_from_a_checkout(self):
        """The README documents it, so it has to keep working."""
        self.assertTrue((REPO / "examples" / "demo_knockdown.py").is_file())
        source = (REPO / "examples" / "demo_knockdown.py").read_text(encoding="utf-8")
        self.assertIn("from cas13guide.demo import", source)


class TestThePackagedDataIsDeclared(unittest.TestCase):
    """The Arc weights are shipped data, not code, so they need declaring.

    A wheel that omitted them would import fine and fail only when somebody
    passed ``--arc``, which is exactly the shape of failure this package is
    arranged against.
    """

    def test_the_weights_are_in_the_package_directory(self):
        weights = sorted((REPO / "src" / "cas13guide" / "models" / "arc").glob("*.onnx"))
        self.assertEqual([w.name for w in weights],
                         ["fold_0.onnx", "fold_1.onnx", "fold_2.onnx"])

    def test_pyproject_declares_them_as_package_data(self):
        import tomllib

        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        declared = config["tool"]["setuptools"]["package-data"]["cas13guide"]
        self.assertIn("models/arc/*.onnx", declared)
        self.assertIn("py.typed", declared)

    def test_every_declared_console_script_resolves(self):
        import importlib
        import tomllib

        config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        for name, target in config["project"]["scripts"].items():
            module_name, _, attribute = target.partition(":")
            with self.subTest(script=name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, attribute)))


if __name__ == "__main__":
    unittest.main()
