"""Tests that cannot fail, hunted as a class rather than one at a time.

Nine of these turned up across nine review passes, and two of the nine were
written *by* the pass that was hunting them. They keep recurring because each
looks correct in review: a green test is indistinguishable from a green test
until you break the thing it names.

Three shapes have actually occurred here, and each is checkable statically:

* a ``__main__`` guard placed above a class, so ``python tests/x.py`` never
  defines it and silently runs a subset -- ``test_blast.py`` ran 21 of its 27
  tests that way, the six missing ones being exactly the class that checks
  BLAST is reachable from the CLI at all. Then it happened twice more, in files
  appended to during later passes;
* a test with no assertion in it at all;
* ``assertTrue`` on a class attribute, which for a ``slots`` dataclass is a
  ``member_descriptor`` whose ``bool()`` is unconditionally True and has no
  relationship to the declared default -- so the test named for that default
  passed whichever way it was set.

This does not catch every vacuous test; nothing static can. It catches the three
that have actually happened, which is the difference between a lesson and a
guard.
"""

import ast
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def iter_tests(tree: ast.AST):
    """Not named ``test_*``: pytest collects module-level functions by prefix,
    and a generator collected as a test is an error rather than a check."""
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test"):
                yield cls, fn


def asserts_something(fn: ast.FunctionDef) -> bool:
    """Any call whose callee mentions ``assert``, or a bare ``assert``.

    Deliberately broad. ``assert_no_regression(report, report)`` raises on
    failure, so the call *is* the assertion, and ``np.testing.assert_array_equal``
    is not a method on ``self`` -- an earlier draft of this check flagged both
    and would have been ignored for crying wolf.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            callee = getattr(node.func, "attr", None) or getattr(node.func, "id", "")
            if "assert" in (callee or "").lower():
                return True
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                callee = getattr(getattr(call, "func", None), "attr", "") or ""
                if "assert" in callee.lower() or "raises" in callee.lower():
                    return True
    return False


MODULES = sorted(TESTS.glob("test_*.py"))


class TestEveryTestModuleRunsWholeUnderDirectExecution(unittest.TestCase):
    """``python tests/x.py`` must run the same tests ``pytest`` collects."""

    def test_nothing_is_defined_after_the_main_guard(self):
        offenders = []
        for path in MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            guard = None
            for node in tree.body:
                if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                        and getattr(node.test.left, "id", "") == "__name__"):
                    guard = node.lineno
            if guard is None:
                continue
            for node in tree.body:
                if (isinstance(node, (ast.ClassDef, ast.FunctionDef))
                        and node.lineno > guard):
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(
            sorted(offenders), [],
            "defined after the __main__ guard, so direct execution skips it",
        )


class TestEveryTestAssertsSomething(unittest.TestCase):
    def test_no_test_function_is_empty_of_assertions(self):
        offenders = [
            f"{path.name}:{fn.lineno} {cls.name}.{fn.name}"
            for path in MODULES
            for cls, fn in iter_tests(ast.parse(path.read_text(encoding="utf-8")))
            if not asserts_something(fn)
        ]
        self.assertEqual(sorted(offenders), [])


class TestNoAssertionOnAClassAttribute(unittest.TestCase):
    """The ``slots`` dataclass trap: read the default off an *instance*."""

    def test_no_asserttrue_on_a_bare_class_attribute(self):
        offenders = []
        for path in MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for _cls, fn in iter_tests(tree):
                for node in ast.walk(fn):
                    if not (isinstance(node, ast.Call)
                            and getattr(node.func, "attr", "") in ("assertTrue", "assertFalse")
                            and node.args):
                        continue
                    arg = node.args[0]
                    if (isinstance(arg, ast.Attribute)
                            and isinstance(arg.value, ast.Name)
                            and arg.value.id[:1].isupper()):
                        offenders.append(
                            f"{path.name}:{node.lineno} {arg.value.id}.{arg.attr}"
                        )
        self.assertEqual(sorted(offenders), [])


class TestThisCheckCoversTheSuite(unittest.TestCase):
    """A guard over an empty set is the very thing being guarded against."""

    def test_it_sees_every_test_module(self):
        self.assertGreater(len(MODULES), 40)

    def test_it_sees_the_test_functions_in_them(self):
        total = sum(
            1
            for path in MODULES
            for _cls, _fn in iter_tests(ast.parse(path.read_text(encoding="utf-8")))
        )
        self.assertGreater(total, 500)


if __name__ == "__main__":
    unittest.main()
