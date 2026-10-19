"""Real localhost socket regressions; no GPU or simulator required."""

import socket
import unittest

from run_task import check_policy_port_available


class PolicyPortTests(unittest.TestCase):
    def listener(self):
        server = socket.socket()
        self.addCleanup(server.close)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        return server

    def test_live_listener_still_rejected(self):
        server = self.listener()
        with self.assertRaises(OSError):
            check_policy_port_available(server.getsockname()[1])

    def test_closed_connection_does_not_skip_next_task(self):
        server = self.listener()
        port = server.getsockname()[1]
        with socket.create_connection(server.getsockname(), timeout=3) as client:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(3)
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                client.shutdown(socket.SHUT_WR)
                self.assertEqual(connection.recv(1), b"")
        server.close()
        # The old probe reproduces the transient false positive.
        with socket.socket() as old_probe:
            with self.assertRaises(OSError):
                old_probe.bind(("127.0.0.1", port))
        check_policy_port_available(port)


if __name__ == "__main__":
    unittest.main()
