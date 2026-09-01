"""Dependencies point one way, and the import graph stays acyclic.

Neither property had a test, and both had already been broken once: a service
reached up into ``cli`` for a shared helper, which made the README's "a real
deployment can delete services/webapp.py and keep everything else" true in only
one direction.

Checked by parsing rather than by importing, so a violation is reported as a
readable list rather than as an ``ImportError`` raised from wherever the cycle
happened to be entered first.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "cas13guide"

#: Intended layering, lowest first. A module may import its own layer or below.
#: ``metrics`` is deliberately absent: ``metrics/bounds.py`` is a primitive that
#: sits below ``models`` while ``metrics/aggregate.py`` consumes it, so the
#: package spans two layers and only the module-level graph is meaningful.
LAYERS = ("util", "contract", "models", "services", "pipeline")


def module_name(path: Path) -> str:
    name = path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def top_level_imports(path: Path) -> set[str]:
    """Relative imports at module scope, resolved to absolute module names.

    Imports inside a function are lazy and deliberate here -- ``workflow`` is
    the composition root and reaches into the pipeline that way on purpose.

    The resolution is the fiddly part and the first version of it was wrong:
    ``from ..contract.enums import X`` inside ``pipeline/b_qc.py`` resolved to
    ``pipeline.contract.enums``, which is in no graph, so the edge vanished.
    Nearly every import in this package is ``..``-level, so the graph came out
    almost empty and the cycle check below could not have failed. For a module
    ``a.b.c``, level 1 means the package ``a.b`` and level *n* strips *n* parts;
    for a package's own ``__init__``, level 1 means that package itself.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = module_name(path).split(".")
    if path.name != "__init__.py":
        parts = parts[:-1]                    # the package containing this module
    found = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            prefix = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
            found.add(".".join([p for p in prefix if p] + ([node.module] if node.module else [])))
    return found


GRAPH = {module_name(p): top_level_imports(p) for p in sorted(ROOT.rglob("*.py"))}


class TestTheImportGraphIsAcyclic(unittest.TestCase):
    """A cycle works until an import order changes, and then it does not."""

    def test_no_module_level_cycle(self):
        colour: dict[str, int] = {}
        cycles: list[list[str]] = []

        def visit(node: str, stack: list[str]) -> None:
            colour[node] = 1
            for dep in sorted(GRAPH.get(node, ())):
                targets = [dep] + [m for m in GRAPH if m.startswith(dep + ".")]
                for target in targets:
                    if target not in GRAPH or target == node:
                        continue
                    if colour.get(target) == 1:
                        cycles.append(stack + [node, target])
                    elif colour.get(target) is None:
                        visit(target, stack + [node])
            colour[node] = 2

        for module in GRAPH:
            if colour.get(module) is None:
                visit(module, [])
        self.assertEqual(cycles, [], "top-level import cycle")

    def test_the_bound_primitive_stays_a_leaf(self):
        """``Interval`` is what ``models`` is built from. If this file grows a
        top-level import of ``models``, the packages really do become cyclic."""
        self.assertEqual(GRAPH["metrics.bounds"], set())


class TestNothingImportsTheEntryPoint(unittest.TestCase):
    """``cli`` composes the package; the package does not reach back into it."""

    def test_no_module_imports_cli(self):
        offenders = sorted(
            name for name, deps in GRAPH.items()
            if name not in ("cli", "__main__") and any(d.endswith("cli") for d in deps)
        )
        self.assertEqual(offenders, [])

    def test_the_wrapper_is_allowed_to(self):
        """``python -m cas13guide`` is exactly what ``__main__`` is for."""
        self.assertIn("cli", GRAPH["__main__"])


class TestLayersPointDownward(unittest.TestCase):
    """At module scope. Lazy imports are the documented escape hatch."""

    def test_no_upward_top_level_import(self):
        rank = {name: index for index, name in enumerate(LAYERS)}
        offenders = []
        for name, deps in GRAPH.items():
            package = name.split(".")[0]
            if package not in rank:
                continue
            for dep in deps:
                target = dep.split(".")[0]
                if target in rank and rank[target] > rank[package]:
                    offenders.append(f"{name} -> {dep}")
        self.assertEqual(sorted(offenders), [])


if __name__ == "__main__":
    unittest.main()
