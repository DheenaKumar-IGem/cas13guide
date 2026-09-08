"""The browser front end (FR1.2, NFR2) and its error behaviour (NFR3).

Runs a real server on an ephemeral port and talks HTTP to it, rather than
calling the handler methods directly: the parts most likely to break are the
multipart parsing and the status codes, and neither is exercised by a unit test
that hands the handler a dict.
"""

import http.client
import re
import threading
import unittest
import uuid

from helpers import *  # noqa: F401,F403

from cas13guide.services.webapp import (
    _parse_multipart,
    _request_from_fields,
    make_server,
)
from cas13guide.services.workflow import DesignRequest

TARGET = (
    "AUGGCACUGAAGUCCAUCGACUGGAACUUCCGAGUGCAUGGCAUUACCGAGUCCUAUGGCAAGUUCGAC"
    "CUGAUCGAGAAGUGCCUGGAAUCCAUGGGCUACCGAGAUCUGGCCAAGUUCGGCAUCUAA"
)


class ServerTestCase(unittest.TestCase):
    """One server for the whole class; each test is a fresh connection."""

    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read().decode()
        finally:
            connection.close()

    def form_post(self, fields, files=None):
        boundary = f"----{uuid.uuid4().hex}"
        parts = []
        for name, value in fields.items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            )
        for name, (filename, content) in (files or {}).items():
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: text/plain\r\n\r\n{content}\r\n"
            )
        body = ("".join(parts) + f"--{boundary}--\r\n").encode()
        return self.request(
            "POST", "/design", body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}",
             "Content-Length": str(len(body))},
        )


class TestFormPage(ServerTestCase):
    def test_the_form_is_served(self):
        status, _headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("<form", body)
        self.assertIn('name="sequence"', body)
        self.assertIn('type="file"', body)          # FR1.2 upload

    def test_defaults_match_the_shared_request_object(self):
        """The form and the CLI must not disagree about what the default is."""
        _status, _headers, body = self.request("GET", "/")
        defaults = DesignRequest()
        for value in (defaults.effector, defaults.assay, defaults.task):
            self.assertIn(f"<option selected>{value}</option>", body)

    def test_every_preset_is_offered(self):
        from cas13guide.contract.presets import preset_ids

        _status, _headers, body = self.request("GET", "/")
        for preset_id in preset_ids():
            self.assertIn(preset_id, body)

    def test_security_headers_are_set(self):
        _status, headers, _body = self.request("GET", "/")
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")

    def test_an_unknown_path_is_a_404(self):
        status, _headers, _body = self.request("GET", "/wat")
        self.assertEqual(status, 404)


class TestDesignRoundTrip(ServerTestCase):
    def test_a_pasted_sequence_produces_a_report(self):
        status, _headers, body = self.form_post({"sequence": TARGET})
        self.assertEqual(status, 200)
        self.assertIn("Cas13 guide design", body)
        self.assertIn("<table id=\"results\"", body)
        self.assertIn("design another", body)

    def test_an_uploaded_fasta_produces_a_report(self):
        """FR1.2 -- the upload path, not just the textarea."""
        status, _headers, body = self.form_post(
            {"sequence": ""},
            files={"fasta": ("t.fa", f">uploaded\n{TARGET}\n")},
        )
        self.assertEqual(status, 200)
        self.assertIn("uploaded", body)

    def test_the_runs_limitations_are_shown_with_the_results(self):
        """A user must not reach a guide table without seeing what was not checked.

        These are derived from the run rather than a fixed banner, so this also
        pins that the derivation reaches the page at all.
        """
        _status, _headers, body = self.form_post({"sequence": TARGET})
        # the calibration this server seeds is synthetic, and it says so
        self.assertIn("SYNTHETIC", body)
        # the built-in activity model is a heuristic, and it says so
        self.assertIn("heuristic", body.lower())
        # and off-target scope is bounded by what was loaded
        self.assertIn("Anything not loaded is", body)

    def test_a_limitation_that_does_not_apply_is_not_shown(self):
        """The point of deriving them: no warning for a model that is not in use."""
        _status, _headers, body = self.form_post(
            {"sequence": TARGET, "assay": "cell_culture_knockdown", "task": "knockdown"}
        )
        # collateral IS a hazard for a knockdown assay, so that one appears ...
        self.assertIn("Collateral", body)
        # ... and the placeholder-signal-constants warning, which is a detection
        # concern only, does not.
        self.assertNotIn("PLACEHOLDER kinetic constants", body)

    def test_a_preset_can_be_designed_against(self):
        status, _headers, body = self.form_post(
            {"sequence": "", "preset": "hsa-miR-21-5p"}
        )
        self.assertEqual(status, 200)
        self.assertIn("hsa-miR-21-5p", body)

    def test_a_preset_run_carries_its_provenance_note(self):
        """The browser user is the one who most needs this.

        A preset is one click and shows no provenance on the page. Someone who
        pastes a sequence owns it; someone who picks from a dropdown is trusting
        us, so whatever qualification applies has to reach the served report and
        not just the terminal.
        """
        _status, _headers, body = self.form_post(
            {"sequence": "", "preset": "hsa-miR-21-5p"}
        )
        self.assertIn("miRBase", body)

    def test_a_pasted_sequence_gets_no_preset_note(self):
        """Both directions -- it is about the preset, not about every run."""
        _status, _headers, body = self.form_post({"sequence": TARGET})
        self.assertNotIn("miRBase", body)


class TestTheServedGuidesAreCorrectAgainstTheInput(ServerTestCase):
    """The browser door proved a table existed. Not that it was right.

    ``examples/verify_output_against_input.py`` re-derives every exported guide
    from the sequence handed to the CLI -- the spacer must be the reverse
    complement of the target site at the coordinate the row itself names -- and
    it does that for 237 guides. Nothing did it for the served page, which
    checked status 200, the presence of ``<table id="results">``, and which
    limitations appeared.

    So the door a bench biologist is most likely to use was the one door whose
    output nothing had checked for correctness. Every assertion below re-derives
    from ``TARGET`` in plain Python, without asking the package how it got there.
    """

    #: One design run for the whole class; the assertions are all views of it.
    _rows = None

    @property
    def rows(self):
        cls = type(self)
        if cls._rows is None:
            _status, _headers, body = self.form_post({"sequence": TARGET})
            cls._rows = self.parse_rows(body)
        return cls._rows

    @staticmethod
    def parse_rows(body):
        """(rank, spacer, site) per served row, as the page renders them."""
        table = re.search(r'<table id="results".*?</table>', body, re.S)
        if not table:
            return []
        rows = []
        for raw in re.findall(r"<tr[^>]*>(.*?)</tr>", table.group(0), re.S)[1:]:
            cells = [re.sub(r"<[^>]+>", "", cell).strip()
                     for cell in re.findall(r"<td[^>]*>(.*?)</td>", raw, re.S)]
            if len(cells) >= 4:
                rows.append({"rank": cells[0], "spacer": cells[1],
                             "site": cells[2], "score": cells[3]})
        return rows

    def test_the_page_serves_guides_at_all(self):
        self.assertGreater(len(self.rows), 10, "fixture must produce a real table")

    def test_every_served_spacer_pairs_with_the_site_its_row_names(self):
        """The one claim a guide row makes that can be checked without trusting
        anything: revcomp(target[start:start+len]) == spacer."""
        from cas13guide.util.seq import revcomp

        for row in self.rows:
            with self.subTest(rank=row["rank"]):
                start = int(row["site"].split(":")[-1])
                site = TARGET[start:start + len(row["spacer"])]
                self.assertEqual(revcomp(site), row["spacer"])

    def test_every_site_lies_inside_the_sequence_that_was_pasted(self):
        """A coordinate past the end would still render, and would still look
        like a coordinate."""
        for row in self.rows:
            with self.subTest(rank=row["rank"]):
                start = int(row["site"].split(":")[-1])
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + len(row["spacer"]), len(TARGET))

    def test_every_served_spacer_is_orderable(self):
        """A browser user copies these into an order form."""
        for row in self.rows:
            with self.subTest(rank=row["rank"]):
                self.assertTrue(set(row["spacer"]) <= set("ACGU"), row["spacer"])

    def test_the_ranks_run_one_to_n_with_no_gaps(self):
        self.assertEqual([r["rank"] for r in self.rows],
                         [str(i) for i in range(1, len(self.rows) + 1)])

    def test_the_scores_do_not_increase_down_the_ranking(self):
        """A ranking that is not ordered is not a ranking, and the reader has
        no way to see it from one screen."""
        scores = [float(r["score"]) for r in self.rows]
        for above, below in zip(scores, scores[1:]):
            self.assertGreaterEqual(above, below)

    def test_the_two_input_doors_answer_the_same_question(self):
        """Pasting a sequence and uploading it as FASTA is the same run.

        Two doors onto one pipeline is exactly where this package has drifted
        before, and the browser has two of them on one form.
        """
        _status, _headers, uploaded = self.form_post(
            {"sequence": ""}, files={"fasta": ("t.fa", f">query\n{TARGET}\n")})
        pasted = [(r["rank"], r["spacer"], r["site"]) for r in self.rows]
        served = [(r["rank"], r["spacer"], r["site"])
                  for r in self.parse_rows(uploaded)]
        self.assertEqual(pasted, served)


class TestErrorsAreLegibleNotTracebacks(ServerTestCase):
    """NFR3 -- malformed input must not return a stack trace to a browser."""

    def test_a_bad_character_returns_400_and_says_where(self):
        status, _headers, body = self.form_post({"sequence": "AUGCXXAUGC"})
        self.assertEqual(status, 400)
        self.assertIn("position 5", body)
        self.assertNotIn("Traceback", body)
        self.assertIn("<form", body)        # and the form comes back to retry

    def test_no_sequence_says_what_to_do(self):
        status, _headers, body = self.form_post({"sequence": "   "})
        self.assertEqual(status, 400)
        self.assertIn("No sequence supplied", body)

    def test_a_non_numeric_field_is_reported_by_name(self):
        status, _headers, body = self.form_post(
            {"sequence": TARGET, "max_mismatches": "lots"}
        )
        self.assertEqual(status, 400)
        self.assertIn("max mismatches", body)
        self.assertNotIn("Traceback", body)

    def test_an_oversized_upload_is_refused_before_it_is_read(self):
        """Send a body that really is too big, against a lowered ceiling.

        The obvious version -- declare a huge Content-Length and send three
        bytes -- is a race: the server answers 413 and closes without reading,
        while the client is still mid-send, and the reset surfaces as a flaky
        failure under load rather than a clean assertion. Lowering the limit
        keeps the test honest about what it sends and fast.
        """
        import cas13guide.services.webapp as webapp

        original = webapp.MAX_UPLOAD_BYTES
        webapp.MAX_UPLOAD_BYTES = 128
        try:
            body = b"sequence=" + b"A" * 512
            status, _headers, _body = self.request(
                "POST", "/design", body,
                {"Content-Type": "application/x-www-form-urlencoded",
                 "Content-Length": str(len(body))},
            )
        finally:
            webapp.MAX_UPLOAD_BYTES = original
        self.assertEqual(status, 413)
        self.assertGreater(original, 1024 * 1024)   # the shipped limit is sane

    def test_a_hostile_sequence_name_is_escaped_in_the_report(self):
        status, _headers, body = self.form_post(
            {"sequence": f">a<script>alert(1)</script>\n{TARGET}"}
        )
        self.assertEqual(status, 200)
        self.assertNotIn("<script>alert(1)</script>", body)
        self.assertIn("&lt;script&gt;", body)


class TestBadRequestsGetAnAnswerNotATraceback(ServerTestCase):
    """Error paths on the door aimed at people who will not use a command
    line. None of them had ever executed.

    Each is the difference between a message and a stack trace. A browser or
    a script gets these wrong routinely -- a stale bookmark, a client posting
    a form the ordinary urlencoded way, a part sent without a name -- and this
    is the surface where a traceback is least recoverable, because the person
    reading it did not choose a terminal.
    """

    CRLF = chr(13) + chr(10)

    def test_a_post_to_the_wrong_path_is_a_404_with_a_page(self):
        status, _headers, body = self.request(
            "POST", "/desgin", body=b"", headers={"Content-Length": "0"})
        self.assertEqual(status, 404)
        self.assertIn("Not found.", body)
        self.assertNotIn("Traceback", body)

    def test_a_query_string_does_not_change_the_route(self):
        """`/design?utm=1` is still the design endpoint, not a 404."""
        status, _headers, body = self.request(
            "POST", "/design?utm=1", body=b"",
            headers={"Content-Length": "0",
                     "Content-Type": "multipart/form-data; boundary=zz"})
        self.assertNotEqual(status, 404, body[:200])

    def test_a_body_that_is_not_multipart_is_refused_legibly(self):
        """An ordinary urlencoded post carries no field this parser reads, so
        the run has no sequence -- and has to say so rather than raise."""
        body = b"sequence=ACGUACGUACGUACGUACGUACGU"
        status, _headers, page = self.request(
            "POST", "/design", body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Content-Length": str(len(body))})
        self.assertNotIn("Traceback", page)
        self.assertIn("sequence", page.lower())

    def test_a_part_without_a_name_is_skipped_not_fatal(self):
        """`get_param` returns nothing for a part with no name, and the field
        dict must not gain a `None` key that nothing downstream looks up."""
        boundary = "----zzz"
        eol = self.CRLF
        body = (
            "--" + boundary + eol
            + "Content-Disposition: form-data" + eol + eol
            + "orphan" + eol
            + "--" + boundary + eol
            + 'Content-Disposition: form-data; name="sequence"' + eol + eol
            + "ACGU" + eol
            + "--" + boundary + "--" + eol
        ).encode()
        fields = _parse_multipart(
            body, "multipart/form-data; boundary=" + boundary)
        # The exact key set, not `assertNotIn(None, ...)`: without the guard
        # the name falls through to `str(raw_name)`, which is the *string*
        # "None" rather than the object, so checking for the object passes
        # against the very code this test exists to reject. The sweep caught
        # that, on a test written minutes earlier.
        self.assertEqual(set(fields), {"sequence"})
        self.assertEqual(fields["sequence"], "ACGU")


class TestMultipartParsing(unittest.TestCase):
    """``cgi`` was removed in 3.13, so this uses ``email``; pin the behaviour."""

    def _body(self, boundary, parts):
        return ("".join(parts) + f"--{boundary}--\r\n").encode()

    def test_fields_and_files_are_separated(self):
        boundary = "abc123"
        body = self._body(boundary, [
            f'--{boundary}\r\nContent-Disposition: form-data; name="effector"\r\n\r\n'
            "LwaCas13a\r\n",
            f'--{boundary}\r\nContent-Disposition: form-data; name="fasta"; '
            'filename="t.fa"\r\n\r\n>x\nAUGC\r\n',
        ])
        fields = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(fields["effector"], "LwaCas13a")
        self.assertIn(">x", fields["fasta__file"])
        self.assertNotIn("fasta", fields)      # the file is under its own key

    def test_an_empty_file_input_is_ignored(self):
        """Browsers send an empty part for a file input nobody touched."""
        boundary = "abc123"
        body = self._body(boundary, [
            f'--{boundary}\r\nContent-Disposition: form-data; name="fasta"; '
            'filename=""\r\n\r\n\r\n',
        ])
        fields = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        self.assertNotIn("fasta__file", fields)

    def test_an_uploaded_file_is_used_when_the_textarea_is_empty(self):
        request = _request_from_fields({"sequence": "", "fasta__file": ">x\nAUGC"})
        self.assertIsNone(request.sequence_text)
        self.assertEqual(request.fasta_text, ">x\nAUGC")

    def test_the_two_are_kept_apart_so_the_one_source_guard_can_see_both(self):
        """This asserted that the paste silently won.

        It pinned the precedence the browser had before ``_read_records``
        began refusing more than one target source -- and because the form
        folded the upload into the paste field, that guard could not see the
        upload at all, so the file was dropped with nothing said. Which is
        exactly the behaviour the guard exists to prevent, on the door it was
        written for.
        """
        request = _request_from_fields({"sequence": "AUGC", "fasta__file": ">x\nGGGG"})
        self.assertEqual(request.sequence_text, "AUGC")
        self.assertEqual(request.fasta_text, ">x\nGGGG")

    def test_paste_plus_upload_is_refused_rather_than_silently_resolved(self):
        from cas13guide.services.workflow import DesignRequestError, run_request

        request = _request_from_fields(
            {"sequence": "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC", "fasta__file": ">x\nGGGG"}
        )
        with self.assertRaises(DesignRequestError) as caught:
            run_request(request)
        self.assertIn("--fasta", str(caught.exception))

    def test_blank_selects_fall_back_to_the_defaults(self):
        request = _request_from_fields({"sequence": "AUGC", "effector": "", "assay": ""})
        defaults = DesignRequest()
        self.assertEqual(request.effector, defaults.effector)
        self.assertEqual(request.assay, defaults.assay)


class TestTheBrowserDoorCanSupplyAnOffTargetCorpus(unittest.TestCase):
    """Without one it cannot certify anything, and that was a regression.

    The universe-adequacy gate reads what the index actually holds, so a
    corpus covering none of the universes an assay is exposed to leaves every
    guide at REVIEW. That is the honest answer -- but the command line can act
    on it with `--background` and the form could not, so a run made through
    the browser could not reach PASS at all.

    It is a capability the gate fix removed rather than one this door never
    had: before it, the shortfall was simply never noticed and the same run
    certified guides against a corpus of one sequence.

    `background_text` mirrors `variants_text`, kept separate from
    `background_path` for the reason that field records -- the two doors must
    reach the same code and a browser has no path to give.
    """

    ASSAY = "cell_culture_knockdown"

    @classmethod
    def setUpClass(cls):
        import random

        rng = random.Random(23)

        def seq(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        cls.target = seq(300)
        eol = chr(10)
        rows = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna")):
            rows += [">" + name + " biotype=" + biotype, seq(300)]
        cls.background = eol.join(rows) + eol

    def _run(self, **extra):
        from cas13guide.services.workflow import run_request

        fields = {
            "sequence": self.target, "assay": self.ASSAY,
            "task": "knockdown", "effector": "RfxCas13d",
        }
        fields.update(extra)
        return run_request(_request_from_fields(fields)).report

    def test_the_form_offers_a_background_upload(self):
        from cas13guide.services.webapp import _form_page

        page = _form_page()
        self.assertIn('name="background"', page)
        self.assertIn('name="background_biotype"', page)

    def test_the_form_says_what_an_empty_background_costs(self):
        """A reader has to know before they submit, not after."""
        from cas13guide.services.webapp import _form_page

        page = _form_page()
        self.assertIn("biotype=rrna", page)
        self.assertIn("REVIEW", page)

    def test_without_a_background_nothing_is_certified(self):
        report = self._run()
        self.assertEqual(report["verdict_counts"]["PASS"], 0)
        gaps = [item for item in report["limitations"]
                if item["id"] == "universe-gap"]
        self.assertTrue(gaps, "no PASS, and no statement of why")

    def test_with_one_the_same_target_certifies(self):
        report = self._run(background__file=self.background)
        self.assertGreater(report["verdict_counts"]["PASS"], 0)
        self.assertEqual(
            [item for item in report["limitations"]
             if item["id"] == "universe-gap"],
            [],
        )

    def test_the_uploaded_records_keep_their_own_biotypes(self):
        """The header labels have to survive the browser path too."""
        report = self._run(background__file=self.background)
        by_locus = {row["locus"]: row["biotype"]
                    for row in report["off_targets"]}
        expected = {"RN18S1": "rrna", "MT-CO1": "mito",
                    "LINC01": "ncrna", "PREM1": "pre_mrna"}
        for locus, biotype in by_locus.items():
            if locus in expected:
                self.assertEqual(biotype, expected[locus], locus)


class TestTheFormCanSkipTheSignalModel(unittest.TestCase):
    """A control the terminal had and this door did not.

    `--no-signal` exists on the CLI. The form posted twelve inputs and this was
    not one of them, so the browser user had no way to decline the
    collateral-signal model -- most of the wall clock on any real target, and
    the source of a column the run reports and then says it does not rank on.

    Both doors are thin wrappers over `services/workflow.run_request`, so the
    capability was always there; only the affordance was missing.
    """

    def _fields(self, **extra):
        base = {
            "sequence": ">T1\n" + ("ACGUUGCAUGGAUCCAUAGGACUCGGUAAUGCCAUG" * 6),
            "preset": "", "background_biotype": "coding",
            "effector": "LwaCas13a", "assay": "in_vitro_detection",
            "task": "detection", "max_mismatches": "4",
            "min_accessibility": "0",
        }
        base.update(extra)
        return base

    def test_the_form_offers_the_control(self):
        from cas13guide.services.webapp import _form_page

        self.assertIn('name="skip_signal"', _form_page())

    def test_leaving_it_unchecked_keeps_the_signal_model(self):
        """An unchecked box posts nothing, so absence must mean the default."""
        from cas13guide.services.webapp import _request_from_fields

        request = _request_from_fields(self._fields())
        self.assertTrue(request.use_signal)

    def test_checking_it_turns_the_model_off(self):
        from cas13guide.services.webapp import _request_from_fields

        request = _request_from_fields(self._fields(skip_signal="on"))
        self.assertFalse(request.use_signal)

    def test_the_choice_reaches_the_report(self):
        """Not just the request object -- the column it controls.

        Measured on a 1 kb target: 20.4 s with the model against 2.6 s
        without, ranking the same 746 guides.
        """
        from cas13guide.services.webapp import _request_from_fields
        from cas13guide.services.workflow import run_request

        with_signal = run_request(_request_from_fields(self._fields())).report
        without = run_request(
            _request_from_fields(self._fields(skip_signal="on"))).report

        self.assertTrue(with_signal["ranked"], "fixture ranked nothing")
        self.assertEqual(len(with_signal["ranked"]), len(without["ranked"]),
                         "skipping the signal model changed which guides rank")
        self.assertIsNotNone(with_signal["ranked"][0]["predicted_signal"])
        self.assertIsNone(without["ranked"][0]["predicted_signal"])


if __name__ == "__main__":
    unittest.main()
