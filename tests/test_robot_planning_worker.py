"""Native planning is warmed in an isolated worker, never in the control process."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from astrabot.robot import collision
from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel
from astrabot.robot.service import _planning_pool


def worker_probe():
    """Report worker identity and exercise an actual geometry query."""
    return (
        os.getpid(),
        collision.native_backend() is not None,
        collision.convex_hulls_within_distance([[0, 0, 0]], [[1, 0, 0]], 0.01),
    )


def model_probe(model, body, hands):
    """Exercise serialized full-body and both TCP chains in the actual worker."""
    return [
        model.poses(body, hands),
        model.pinch_poses(body, hands),
        *[model._poses(body, hands, model._tcp_orders[side]) for side in ("left", "right")],
    ]


class PlanningWorkerTests(unittest.TestCase):
    def test_serialized_model_matches_reference_in_accelerated_worker(self):
        model = RobotModel(Config())
        body = np.linspace(-0.1, 0.1, 29)
        hands = np.linspace(80, 240, 12)
        with patch.object(collision, "_FAST", None):
            expected = model_probe(model, body, hands)
        with _planning_pool() as pool:
            for _ in range(2):
                actual = pool.submit(model_probe, model, body, hands).result(timeout=90)
                for expected_chain, actual_chain in zip(expected, actual):
                    self.assertEqual(expected_chain.keys(), actual_chain.keys())
                    for frame in expected_chain:
                        np.testing.assert_allclose(actual_chain[frame], expected_chain[frame], atol=1e-12)

    def test_worker_initializes_once_and_parent_keeps_reference_backend(self):
        with patch.object(collision, "_FAST", None), _planning_pool() as pool:
            first = pool.submit(worker_probe).result(timeout=90)
            second = pool.submit(worker_probe).result(timeout=10)
            self.assertNotEqual(first[0], os.getpid())
            self.assertEqual(first, second)
            self.assertFalse(first[2])
            self.assertIsNone(collision.native_backend())
            try:
                import numba  # noqa: F401 -- optional dependency availability
            except ImportError:
                self.assertFalse(first[1])
            else:
                self.assertTrue(first[1])


if __name__ == "__main__":
    unittest.main()
