"""One slow design must not take the whole server down with it.

`http.server.HTTPServer` handles one request at a time. A design on a 500 nt
target takes seconds, and for that whole window every other connection is
refused outright rather than queued -- a second tab, a reload, even the
favicon. Found by driving the real page in a browser: the submit went out, the
server took the request, and a concurrent connection got ECONNREFUSED rather
than waiting.

That is a bad failure on this interface specifically. `serve` exists for people
who will not use a command line (NFR2), and the page itself says "the run
happens in this request, so the page will wait" -- so waiting is the expected
behaviour and an impatient reload is the expected user action. Under the
single-threaded server that reload killed the pending run and returned nothing.

These tests pin the fix at the level that matters: not "is the class name
right", but "does a second request get served while a first is still running".
"""

import http.client
import threading
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.webapp import make_server


class ServerFixture(unittest.TestCase):
    def setUp(self):
        self.server = make_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=30)

    def get(self, path="/", timeout=60):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()


class TestConcurrentRequests(ServerFixture):
    def test_the_server_is_threaded(self):
        """The property, stated directly."""
        from socketserver import ThreadingMixIn

        self.assertIsInstance(self.server, ThreadingMixIn)

    def test_threads_do_not_outlive_the_process(self):
        """daemon_threads, so Ctrl-C is not held open by an in-flight design."""
        self.assertTrue(getattr(self.server, "daemon_threads", False))

    def test_a_get_is_served_while_a_real_design_is_in_flight(self):
        """The invariant, driven by something that actually occupies a thread.

        An earlier version of this test held a client-side event instead of the
        server and passed under BOTH servers -- fast requests simply queue in
        the listen backlog, so nothing was being measured. It takes a genuinely
        slow *server-side* request to tell the two apart, which is what a design
        POST is.
        """
        import time

        target = ("AGAUCCCUAAAGACACGUCGGGCAGAGGCUGAGGCCACCACCCGCCAUUAACCUCUACC"
                  "GCGCCCUGGAAUCCGUGCCCCAGGCACGCUCCAGCGAUCUCAAACAUUAAGGUAGCUUC"
                  "UGACUUGCCUUCAACAAGGAACGGCGACCUUUCGCUACAUUAACUCAUUUUUCGCCACG"
                  "UGUGCCCGCUGCCAGAUUAACGCUAGUGUAGUGGUAGACUAAGUAGGAGGUGGAACGCA"
                  "AAUAGAAGGCACGUCCAGGCGGACAGGGAGCGAGUAAUAAAUGGACUGCCCUCUUUACG")
        boundary = "----cas13guideconcurrency"
        crlf = chr(13) + chr(10)
        body = crlf.join([
            f"--{boundary}",
            'Content-Disposition: form-data; name="sequence"',
            "",
            target,
            f"--{boundary}--",
            "",
        ]).encode()

        design_finished = threading.Event()
        design_status = []

        def design():
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
            try:
                conn.request(
                    "POST", "/design", body,
                    {"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Content-Length": str(len(body))},
                )
                design_status.append(conn.getresponse().status)
            except Exception as exc:
                design_status.append(repr(exc))
            finally:
                conn.close()
                design_finished.set()

        worker = threading.Thread(target=design, daemon=True)
        worker.start()

        # give the design time to be picked up and start occupying a thread
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not design_finished.is_set():
            time.sleep(0.05)

        if design_finished.is_set():
            self.skipTest("the design finished too fast to race against")

        status, _body = self.get(timeout=30)
        self.assertEqual(
            status, 200, "a plain GET was not served while a design was running"
        )
        self.assertFalse(
            design_finished.is_set(),
            "the design completed first, so nothing was actually overlapped",
        )

        worker.join(timeout=300)
        self.assertEqual(design_status, [200])
