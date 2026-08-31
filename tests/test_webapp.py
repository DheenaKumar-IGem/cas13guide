"""The browser front end (FR1.2, NFR2) and its error behaviour (NFR3).

Runs a real server on an ephemeral port and talks HTTP to it, rather than
calling the handler methods directly: the parts most likely to break are the
multipart parsing and the status codes, and neither is exercised by a unit test
that hands the handler a dict.
"""

import http.client
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


if __name__ == "__main__":
    unittest.main()
