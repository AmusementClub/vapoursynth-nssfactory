import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import c4_integration_pmu as pmu


class PerfControlTests(unittest.TestCase):
    def test_fifo_acknowledgements_with_and_without_c_terminator(self):
        for payload in (b'ack\n', b'ack\n\x00'):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as td:
                ctl, ack = Path(td) / 'ctl', Path(td) / 'ack'
                os.mkfifo(ctl)
                os.mkfifo(ack)
                commands = []

                def server():
                    c, a = os.open(ctl, os.O_RDWR), os.open(ack, os.O_RDWR)
                    try:
                        for _ in range(2):
                            commands.append(os.read(c, 128))
                            os.write(a, payload)
                    finally:
                        os.close(c)
                        os.close(a)

                def worker(plugin, config):
                    before = pmu.integration.time.perf_counter()
                    after = pmu.integration.time.perf_counter()
                    return dict(ms=(after - before) * 1000)

                thread = threading.Thread(target=server, daemon=True)
                thread.start()
                with patch.object(pmu.integration, 'worker', side_effect=worker):
                    result = pmu.run('fixture', {}, str(ctl), str(ack))
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(commands, [b'enable\n', b'disable\n'])
                self.assertEqual(result['pmu_boundary'], 'frame_requests_only')


if __name__ == '__main__':
    unittest.main()
