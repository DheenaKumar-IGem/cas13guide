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
single-threaded server that reload got nothing at all.

Two earlier versions of the behavioural test here were themselves broken, which
is worth recording because both looked fine:

* the first held a client-side event and asserted eight concurrent GETs all
  returned 200. It passed under BOTH servers -- fast requests simply queue in
  the listen backlog, so it measured nothing;
* the second raced a real design POST, and skipped itself when the design
  finished first. A test that quietly stops testing is better than the first
  only in that it says so.

The version here holds a socket open with a request it never terminates. A
single-threaded server blocks in readline() on that connection and never
returns to accept(); a threaded one serves everyone anyway. No timing
assumptions, nothing to skip.
"""

import http.client
import socket
import threading
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.webapp import make_server

PARTIAL_REQUEST = b"GET / HTTP/1.1\r\nHost: localhost\r\n"
END_OF_HEADERS = b"\r\n"


class ServerFixture(unittest.TestCase):
    def setUp(self):
        self.server = make_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.held: list[socket.socket] = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        # Held connections must be closed BEFORE shutdown, not by addCleanup,
        # which runs after tearDown. A single-threaded server is still blocked
        # in readline() on them, so shutdown() would wait forever and the
        # failure would present as a hung suite rather than a failing test.
        # Whoever reverts the threading fix should get a red run, not a stall.
        for connection in self.held:
            try:
                connection.close()
            except OSError:
                pass
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

    def hold_connection_open(self):
        """A connection with an unterminated request, which the server waits on."""
        held = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        held.sendall(PARTIAL_REQUEST)
        self.held.append(held)
        return held


class TestConcurrentRequests(ServerFixture):
    def test_the_server_is_threaded(self):
        """The property, stated directly."""
        from socketserver import ThreadingMixIn

        self.assertIsInstance(self.server, ThreadingMixIn)

    def test_threads_do_not_outlive_the_process(self):
        """daemon_threads, so Ctrl-C is not held open by an in-flight design."""
        self.assertTrue(getattr(self.server, "daemon_threads", False))

    def test_a_request_is_served_while_another_connection_is_held(self):
        """The behaviour the class name is only a proxy for."""
        self.hold_connection_open()

        status, _body = self.get(timeout=15)
        self.assertEqual(
            status, 200, "a second request was not served while one was in flight"
        )

    def test_several_held_connections_do_not_starve_the_server(self):
        """One stuck client is a nuisance; four must not be an outage."""
        for _ in range(4):
            self.hold_connection_open()

        self.assertEqual(self.get(timeout=15)[0], 200)

    def test_a_held_connection_is_still_answered_once_completed(self):
        """Holding one open must not corrupt it: finish it and it responds."""
        held = self.hold_connection_open()
        self.assertEqual(self.get(timeout=15)[0], 200)

        held.sendall(END_OF_HEADERS)
        self.assertTrue(held.recv(64).startswith(b"HTTP/1."))
