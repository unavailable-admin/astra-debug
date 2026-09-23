"""Garbage collection isolation without modifying the test runner's GC policy."""

import json
import subprocess
import sys
import unittest


class ControlGCTests(unittest.TestCase):
    def test_initialized_graph_is_frozen_but_new_cycles_are_collected(self):
        script = """
import gc, json, weakref
from astrabot.robot.control_gc import ControlGC
class Cycle:
    pass
old = Cycle()
old.reference = old
control = ControlGC()
new = Cycle()
new.reference = new
reference = weakref.ref(new)
del new
gc.collect()
result = {
    'old_frozen': not any(item is old for item in gc.get_objects()),
    'new_collected': reference() is None,
    'automatic_gc_enabled': gc.isenabled(),
    'diagnostics': control.snapshot(),
}
control.close()
result['old_unfrozen'] = any(item is old for item in gc.get_objects())
result['callback_removed'] = control.observe not in gc.callbacks
print(json.dumps(result))
"""
        result = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True, timeout=10))
        for key in ("old_frozen", "new_collected", "automatic_gc_enabled", "old_unfrozen", "callback_removed"):
            self.assertTrue(result[key], key)
        self.assertGreaterEqual(result["diagnostics"]["count"], 1)
        self.assertEqual(result["diagnostics"]["last"]["generation"], 2)
