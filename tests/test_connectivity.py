# tests/test_connectivity.py
"""Unit tests for connectivity.py — the never-blocking "are we online?" flag.

Probes only ever target 127.0.0.1 (a local listening socket for "online", a
closed port for "offline"); no real network is touched:

    python -m unittest discover tests -v
"""

import os
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import connectivity  # noqa: E402


class TestConnectivity(unittest.TestCase):
    def setUp(self):
        self._orig = (connectivity.PROBE_HOST, connectivity.PROBE_PORT,
                      connectivity.PROBE_TTL, connectivity._probe)
        connectivity._online = None
        connectivity._checked_at = None
        connectivity._thread = None
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()

    def tearDown(self):
        connectivity.wait(5)
        (connectivity.PROBE_HOST, connectivity.PROBE_PORT,
         connectivity.PROBE_TTL, connectivity._probe) = self._orig
        connectivity._online = None
        connectivity._checked_at = None
        connectivity._thread = None
        self.listener.close()

    def test_unknown_first_then_online(self):
        connectivity.PROBE_HOST, connectivity.PROBE_PORT = self.listener.getsockname()
        self.assertIsNone(connectivity.state())   # first call only starts a probe
        connectivity.wait(5)
        self.assertTrue(connectivity.state())

    def test_closed_port_is_offline(self):
        connectivity.PROBE_HOST, connectivity.PROBE_PORT = "127.0.0.1", 1
        connectivity.state()
        connectivity.wait(5)
        self.assertFalse(connectivity.state())

    def test_state_never_blocks_on_a_slow_probe(self):
        started = threading.Event()

        def slow_probe():
            started.set()
            time.sleep(1.0)

        connectivity._probe = slow_probe
        t0 = time.monotonic()
        result = connectivity.state()
        elapsed = time.monotonic() - t0
        self.assertTrue(started.wait(2))
        self.assertIsNone(result)
        self.assertLess(elapsed, 0.2)

    def test_mark_offline_holds_until_the_next_probe(self):
        connectivity.PROBE_HOST, connectivity.PROBE_PORT = self.listener.getsockname()
        connectivity.state()
        connectivity.wait(5)
        self.assertTrue(connectivity.state())
        probe_before = connectivity._thread
        connectivity.mark_offline()
        self.assertFalse(connectivity.state())
        self.assertIs(connectivity._thread, probe_before)   # no re-probe yet
        connectivity.PROBE_TTL = 0.0                         # now it's stale
        connectivity.state()
        connectivity.wait(5)
        self.assertTrue(connectivity.state())


if __name__ == "__main__":
    unittest.main()
