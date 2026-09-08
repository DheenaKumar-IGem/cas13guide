"""One default per option, however you arrive at it.

There are two doors into a design and each declares its own defaults. The web
form reads them from ``DesignRequest()`` directly, which is why it cannot
drift. The CLI does not: every ``add_argument`` carries its own ``default=``,
independently written, and nothing compared the two.

Eighteen options are declared in both places. They agree today. Nothing made
them agree, and a run with no flags is the most common run there is -- so a
drift here would change what "the defaults" mean on one door only, silently,
for every user who never set the option.

This is the same shape as the shortfall recorded twice, as prose and as a
reason code on adjacent lines: two representations of one fact, both correct
now, with nothing holding them together.
"""

import dataclasses
import unittest

from helpers import *  # noqa: F401,F403

#: CLI flags that are the negation of a request field. ``--no-structure``
#: exists because structure is on by default and a flag that turns something
#: *off* reads better than one that turns it on; the request field states the
#: positive. Both are right, and they still have to agree.
NEGATED = {"no_structure": "use_structure", "no_signal": "use_signal"}

#: Flags whose name differs from the field they feed.
RENAMED = {"arc": "use_arc"}


def _cli_defaults() -> dict:
    """What ``design`` puts in its namespace when given only a sequence."""
    import cas13guide.cli as cli

    captured: dict = {}

    class Stop(Exception):
        pass

    def capture(args):
        captured.update(vars(args))
        raise Stop()

    original = cli._design
    cli._design = capture
    try:
        cli.main(["design", "--sequence", "ACGU"])
    except Stop:
        pass
    finally:
        cli._design = original
    return captured


class TestBothDoorsMeanTheSameThingByDefault(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cas13guide.services.workflow import DesignRequest

        cls.cli = _cli_defaults()
        request = DesignRequest()
        cls.request = {
            field.name: getattr(request, field.name)
            for field in dataclasses.fields(request)
        }

    def test_the_capture_actually_captured_something(self):
        """Every assertion below iterates a dict. An empty one passes them all."""
        self.assertGreater(len(self.cli), 15)
        self.assertGreater(len(self.request), 15)

    def test_the_shared_options_agree(self):
        shared = sorted(set(self.cli) & set(self.request))
        self.assertGreater(len(shared), 15, f"only {len(shared)} options compared")
        for name in shared:
            with self.subTest(option=name):
                self.assertEqual(
                    self.cli[name], self.request[name],
                    f"`--{name.replace('_', '-')}` defaults to "
                    f"{self.cli[name]!r} on the command line and "
                    f"{self.request[name]!r} in the request the web form reads",
                )

    def test_the_negated_flags_agree_too(self):
        """``--no-structure`` defaulting to False must mean ``use_structure``
        defaults to True, or one door quietly runs without ViennaRNA."""
        for flag, field in NEGATED.items():
            with self.subTest(flag=flag):
                self.assertIn(flag, self.cli)
                self.assertIn(field, self.request)
                self.assertEqual(not self.cli[flag], self.request[field])

    def test_the_renamed_flags_agree_too(self):
        for flag, field in RENAMED.items():
            with self.subTest(flag=flag):
                self.assertIn(flag, self.cli)
                self.assertEqual(self.cli[flag], self.request[field])

    def test_the_web_form_reads_its_defaults_rather_than_restating_them(self):
        """The reason the form cannot drift, pinned so it stays the reason."""
        import inspect

        from cas13guide.services import webapp

        source = inspect.getsource(webapp)
        self.assertIn("defaults = DesignRequest()", source)


if __name__ == "__main__":
    unittest.main()
