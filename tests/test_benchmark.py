"""Exercise the reset/run subprocess interface without touching a simulator."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from astrabot import benchmark


class Benchmark(unittest.TestCase):
    def test_fifty_trials_use_new_entrypoint_and_resume_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "batch"
            commands = []

            def command(args, log, timeout):
                commands.append(args)
                self.assertEqual(args[1:3], ["-m", "astrabot"])
                self.assertEqual(args[args.index("--uri") + 1], "ws://unused:8082")
                path = Path(args[args.index("--output") + 1])
                if args[3] == "reset":
                    path.write_text('{"action_layout":"full36"}')
                else:
                    self.assertEqual(args[3], "run")
                    path.mkdir()
                    (path / "report.json").write_text(
                        json.dumps(
                            {
                                "success_verified": True,
                                "simulation_seconds": 100,
                                "skills_completed": 3,
                                "events": [],
                                "reason": "mocked_success",
                                "video": "",
                            }
                        )
                    )
                return 0

            argv = ["--output", str(out), "--uri", "ws://unused:8082"]
            with patch.object(benchmark, "command", side_effect=command), patch("builtins.print"):
                benchmark.main(argv)
            self.assertEqual([c[3] for c in commands], ["reset", "run"] * 50)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["completed"], 50)
            self.assertEqual(
                Counter(r["speed"] for r in summary["trials"]), {s: 10 for s in benchmark.SPEEDS}
            )
            with patch.object(benchmark, "command") as call, patch("builtins.print"):
                benchmark.main(argv)
                call.assert_not_called()
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["source_hashes"] = {}
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                benchmark.main(argv)
