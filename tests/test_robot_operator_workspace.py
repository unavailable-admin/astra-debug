"""Operator-cleared experiment: retain target/table checks without obstacle certification."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.config import Config, file_digest
from astrabot.robot.scene_builder import SceneBuilder, fit_table, target_height_check
from astrabot.robot.scene_coverage import check_observed_box
from astrabot.robot.scene_identification import annotation_from_decision
from astrabot.robot.table import cube_penetrates
from astrabot.robot.trial_observer import TrialObserver
from tests.test_robot_scene_identification import DIMENSIONS, valid_decision


def decision():
    """One clear A; surrounding glyphs/objects deliberately not certified."""
    result = valid_decision(fully_visible=True)
    result.update(
        target_id=1,
        cube_ids=[1, 2, 3, 4],
        target_verified=True,
        target_upright=True,
        unique_target=False,
        all_cubes_upright=False,
        selection_complete=False,
        selection_confidence=0.1,
        tabletop_clear_except_cubes=False,
        target_uncertainty_reasons=[],
        table_uncertainty_reasons=[],
        uncertainty_reasons=["Other glyphs obscured; cables and nearby objects not checked."],
        unresolved_cube_ids=[4],
        near_edge_normalized=[[x * 1000 / 1919, 0] for x in (50, 400, 800, 1100)],
        left_edge_normalized=[[0, y * 1000 / 1199] for y in (50, 200, 400, 550)],
    )
    return result


class OperatorWorkspaceTests(unittest.IsolatedAsyncioTestCase):

    def test_recorded_raised_samples_choose_stable_consensus_before_minimum_error(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_raised_plane_sampling.json").read_text())
        diagnostics = {}
        plane, residual, inliers = fit_table(
            fixture["letters"], operator_cleared_workspace=True, diagnostics=diagnostics
        )
        self.assertEqual(inliers, {"cube_012", "A", "cube_032", "cube_034"})
        self.assertAlmostEqual(diagnostics["narrow_spread_mm"], 93.34318646, places=6)
        self.assertGreater(diagnostics["rejected_unstable_candidates"], 0)
        self.assertLess(np.linalg.norm(plane[:2]), 0.2)
        for name, error in zip(fixture["letters"], residual):
            if name in inliers:
                self.assertLessEqual(abs(error), 0.005)
        self.assertTrue(target_height_check(fixture["letters"]["A"], plane)["passed"])

    def test_no_alternative_plane_keeps_distribution_and_tilt_limits(self):
        narrow = {
            f"p{i}": {"top_center_xyz": [x, y, z]}
            for i, (x, y, z) in enumerate(
                ((0, 0, 1.04), (0.3, 0.001, 1.04), (0.6, 0.002, 1.04), (0.9, 0.003, 1.04), (0.4, 0.3, 1.3))
            )
        }
        with self.assertRaises(ValueError):
            fit_table(narrow, operator_cleared_workspace=True)
        tilted = {
            f"p{i}": {"top_center_xyz": [x, y, 1.04 + 0.25 * x]}
            for i, (x, y) in enumerate(((0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)))
        }
        with self.assertRaisesRegex(ValueError, "table_tilt_exceeds"):
            fit_table(tilted, operator_cleared_workspace=True)

    def test_recorded_metric_caveats_proceed_to_local_validation(self):
        response = json.loads((Path(__file__).parent / "fixtures/robot_measurement_notes.json").read_text())
        original = copy.deepcopy(response)
        annotation = annotation_from_decision(
            response,
            [{"id": i} for i in range(1, 23)],
            (1200, 1920, 3),
            DIMENSIONS,
            operator_cleared_workspace=True,
        )
        self.assertEqual(annotation["target_id"], 13)
        self.assertEqual(annotation["measurement_notes"], response["table_uncertainty_reasons"])
        self.assertEqual(response, original)
        # A metric advisory must never mask a real visual blocker in the same response.
        for field, value in (
            ("table_edge_visible", False),
            ("table_edge_confidence", 0.5),
            ("target_uncertainty_reasons", response["table_uncertainty_reasons"]),
            ("table_uncertainty_reasons", response["table_uncertainty_reasons"] + ["Wrong edge"]),
            ("table_uncertainty_reasons", [response["table_uncertainty_reasons"][0] + " The top edge is hidden."]),
        ):
            with self.subTest(field=field):
                invalid = dict(response, **{field: value})
                with self.assertRaises(ValueError):
                    annotation_from_decision(
                        invalid,
                        [{"id": i} for i in range(1, 23)],
                        (1200, 1920, 3),
                        DIMENSIONS,
                        operator_cleared_workspace=True,
                    )

    def test_measurement_notes_are_informational_but_do_not_override_failures(self):
        response = decision()
        response["measurement_notes"] = ["Stereo height and edge fit pending."]
        candidates = [{"id": i} for i in range(1, 5)]
        annotation = annotation_from_decision(
            response, candidates, (1200, 1920, 3), DIMENSIONS, operator_cleared_workspace=True
        )
        self.assertEqual(annotation["measurement_notes"], response["measurement_notes"])
        for value in (None, "pending", [None], [""]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid_measurement_notes"):
                annotation_from_decision(
                    dict(response, measurement_notes=value),
                    candidates,
                    (1200, 1920, 3),
                    DIMENSIONS,
                    operator_cleared_workspace=True,
                )
        response["table_uncertainty_reasons"] = ["Wrong edge"]
        with self.assertRaisesRegex(ValueError, "Wrong edge") as raised:
            annotation_from_decision(response, candidates, (1200, 1920, 3), DIMENSIONS, operator_cleared_workspace=True)
        self.assertNotIn("部分积木", str(raised.exception))
        response["unresolved_cube_ids"] = [1, 4]
        with self.assertRaisesRegex(ValueError, "unresolved_cube_ids=1\\n") as raised:
            annotation_from_decision(response, candidates, (1200, 1920, 3), DIMENSIONS, operator_cleared_workspace=True)
        self.assertIn("选定 A", str(raised.exception))

    def test_operator_table_samples_use_stable_majority_and_keep_metric_limits(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_table_sample_outliers.json").read_text())
        with self.assertRaisesRegex(ValueError, "没有足够一致的桌面深度"):
            fit_table(fixture["letters"])
        plane, residual, inliers = fit_table(fixture["letters"], operator_cleared_workspace=True)
        self.assertEqual(inliers, {"cube_004", "cube_006", "A", "cube_016", "cube_019"})
        for name, error in zip(fixture["letters"], residual):
            if name in inliers:
                self.assertLessEqual(abs(error), 0.005)
        self.assertTrue(target_height_check(fixture["letters"]["A"], plane)["passed"])

    def test_operator_table_still_requires_four_points_and_strict_majority(self):
        samples = {
            f"p{i}": {"top_center_xyz": [x, y, z]}
            for i, (x, y, z) in enumerate(
                (x, y, z) for z in (1.04, 1.24) for x, y in ((0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4))
            )
        }
        with self.assertRaisesRegex(ValueError, "没有足够一致的桌面深度"):
            fit_table(samples, operator_cleared_workspace=True)
        with self.assertRaisesRegex(ValueError, "至少需要四块"):
            fit_table(dict(list(samples.items())[:3]), operator_cleared_workspace=True)

    def test_recorded_A_outside_plane_inliers_passes_8mm_height_check(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_target_height.json").read_text())
        plane, _, inliers = fit_table(fixture["letters"])
        self.assertNotIn("A", inliers)
        result = target_height_check(fixture["letters"]["A"], plane)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["error_mm"], -4.981530925, places=6)
        self.assertEqual(result["limit_mm"], 8)

    def test_target_height_accepts_both_8mm_boundaries_and_rejects_beyond(self):
        for error_mm in (-8.01, -8.0, 0, 8.0, 8.01):
            with self.subTest(error_mm=error_mm):
                result = target_height_check({"top_center_xyz": [0.4, 0.2, 1.04 + error_mm / 1000]}, [0, 0, 1])
                self.assertEqual(result["passed"], abs(error_mm) <= 8)

    def test_non_inlier_target_is_never_replaced_by_plane_in_either_mode(self):
        for cleared in (False, True):
            with self.subTest(cleared=cleared), tempfile.TemporaryDirectory() as directory:
                builder, measured = self.make_builder(Path(directory), cleared)
                for field in ("top_center_xyz", "estimated_xyz", "grasp_center_xyz"):
                    measured["letters"]["A"][field][2] -= 0.005
                measured["letters"]["cube_005"] = copy.deepcopy(measured["letters"]["cube_002"])
                builder.candidates.append({"id": 5, "pixel": [400, 300]})
                response = valid_decision()
                response.update(
                    target_id=1,
                    cube_ids=[1, 2, 3, 4, 5],
                    target_verified=True,
                    target_upright=True,
                    target_uncertainty_reasons=[],
                    table_uncertainty_reasons=[],
                )
                response.update({k: v for k, v in decision().items() if k.endswith("_edge_normalized")})
                annotation = annotation_from_decision(
                    response, builder.candidates, builder.left.shape, DIMENSIONS, operator_cleared_workspace=cleared
                )
                with (
                    patch("astrabot.robot.scene_builder.StereoTracker") as tracker,
                    patch(
                        "astrabot.robot.scene_builder.fit_table",
                        return_value=(
                            np.array([0, 0, 1]),
                            np.array([-0.005, 0, 0, 0, 0]),
                            set(measured["letters"]) - {"A"},
                        ),
                    ),
                ):
                    tracker.return_value.locate_images.return_value = measured
                    builder.build(annotation)
                np.testing.assert_array_equal(
                    builder.draft["letters"]["A"]["top_center_xyz"], measured["letters"]["A"]["top_center_xyz"]
                )
                self.assertNotIn("A", builder.draft["modeled_obstacles"])
                self.assertTrue(builder.draft["target_height_check"]["passed"])

    def make_builder(self, root, cleared):
        """Use synthetic stereo measurements with real plane fitting and export."""
        builder = SceneBuilder.__new__(SceneBuilder)
        builder.output = root
        builder.capture_dir = root
        builder.config = Config(scene_measured=True)
        builder.operator_cleared_workspace = cleared
        builder.exported = False
        builder.verify_sources = Mock()
        builder.left = builder.right = np.zeros((1200, 1920, 3), np.uint8)
        intrinsic = np.diag([1000.0, 1000.0, 1.0])
        builder.geometry = SimpleNamespace(
            K_left=intrinsic,
            K_right=intrinsic,
            D_left=np.zeros(5),
            D_right=np.zeros(5),
            R=np.eye(3),
            T=np.array([-0.08, 0, 0]),
            image_size_wh=(1920, 1200),
        )
        builder.transform = np.eye(4)
        builder.meta = {
            "body_after": {"body": [0.0] * 29, "hands": [255.0] * 12, "session": "offline", "generation": 1}
        }
        builder.source_hash = builder.config_hash = builder.calibration_hash = "synthetic"
        letters = {}
        builder.candidates = []
        for index, (x, y) in enumerate(((0.2, 0.2), (0.7, 0.2), (0.2, 0.45), (0.7, 0.45)), 1):
            pixel = [1000 * x / 1.04, 1000 * y / 1.04]
            builder.candidates.append({"id": index, "pixel": pixel})
            letters["A" if index == 1 else f"cube_{index:03d}"] = {
                "pixel": pixel,
                "estimated_xyz": [x, y, 1.02],
                "top_center_xyz": [x, y, 1.04],
                "grasp_center_xyz": [x, y, 1.035],
            }
        return builder, {"letters": letters, "errors": {}}

    def build_scene(self, root, cleared):
        builder, measured = self.make_builder(root, cleared)
        response = decision()
        if cleared:
            response["measurement_notes"] = ["Metric geometry must be checked locally."]
        if not cleared:
            response.update(
                unique_target=True,
                all_cubes_upright=True,
                selection_complete=True,
                selection_confidence=0.99,
                tabletop_clear_except_cubes=True,
                uncertainty_reasons=[],
                unresolved_cube_ids=[],
            )
        path = root / "identification.json"
        path.write_text(json.dumps(response))
        annotation = annotation_from_decision(
            response, builder.candidates, builder.left.shape, DIMENSIONS, operator_cleared_workspace=cleared
        )
        with patch("astrabot.robot.scene_builder.StereoTracker") as tracker:
            tracker.return_value.locate_images.return_value = measured
            builder.build(annotation, identification={"path": str(path), "sha256": file_digest(path)})
        builder.export_automatic()
        return builder

    def test_non_target_blockers_are_ignored_only_with_explicit_operator_policy(self):
        candidates = [{"id": i} for i in range(1, 5)]
        response = decision()
        with self.assertRaisesRegex(ValueError, "selection_confidence|unique_target"):
            annotation_from_decision(response, candidates, (1200, 1920, 3), DIMENSIONS)
        annotation = annotation_from_decision(
            response, candidates, (1200, 1920, 3), DIMENSIONS, operator_cleared_workspace=True
        )
        self.assertEqual(annotation["workspace_policy"], "operator_cleared")
        self.assertEqual(annotation["target_id"], 1)

    def test_target_and_table_uncertainty_still_block(self):
        for field, value in (
            ("target_verified", False),
            ("target_upright", False),
            ("target_confidence", 0.5),
            ("target_uncertainty_reasons", ["A hidden"]),
            ("table_uncertainty_reasons", ["Wrong edge"]),
            ("target_uncertainty_reasons", None),
            ("unresolved_cube_ids", [1]),
        ):
            with self.subTest(field=field):
                response = decision()
                response[field] = value
                with self.assertRaises(ValueError):
                    annotation_from_decision(
                        response,
                        [{"id": i} for i in range(1, 5)],
                        (1200, 1920, 3),
                        DIMENSIONS,
                        operator_cleared_workspace=True,
                    )

    def test_measurement_notes_do_not_skip_local_target_height_check(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, measured = self.make_builder(Path(directory), True)
            response = decision()
            response["measurement_notes"] = ["A height cannot be measured from one image."]
            annotation = annotation_from_decision(
                response, builder.candidates, builder.left.shape, DIMENSIONS, operator_cleared_workspace=True
            )
            measured["letters"]["A"]["top_center_xyz"][2] += 0.020
            with (
                patch("astrabot.robot.scene_builder.StereoTracker") as tracker,
                patch(
                    "astrabot.robot.scene_builder.fit_table",
                    return_value=(np.array([0, 0, 1]), np.array([0.02, 0, 0, 0]), set(measured["letters"]) - {"A"}),
                ),
            ):
                tracker.return_value.locate_images.return_value = measured
                with self.assertRaisesRegex(ValueError, "A 顶面应在桌面上方 40 mm；实测 60.00 mm"):
                    builder.build(annotation)
            self.assertFalse(builder.exported)

    def test_real_export_keeps_table_but_omits_non_target_obstacles_and_tracking(self):
        for cleared in (False, True):
            with self.subTest(cleared=cleared), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                builder = self.build_scene(root, cleared)
                source = json.loads((root / "scene-input.json").read_text())
                manifest = json.loads((root / "export.json").read_text())
                letters = json.loads((root / "scene-letters.json").read_text())["letters"]
                config = json.loads((root / "scene-config.json").read_text())
                self.assertEqual(source["workspace_policy"], "operator_cleared" if cleared else "vision_checked")
                self.assertEqual(len(source["obstacles"]), 0 if cleared else 3)
                self.assertEqual(set(letters), {"A"} if cleared else {"A", "cube_002", "cube_003", "cube_004"})
                self.assertIs(source["scene_complete"], not cleared)
                self.assertIs(manifest["scene_complete"], not cleared)
                self.assertIsNotNone(config["table_plane"])
                self.assertIsNotNone(config["table_footprint"])
                self.assertIsNotNone(config["table_observed"])
                if cleared:
                    self.assertEqual(len(builder.draft["table_support_letters"]), 4)
                    self.assertEqual(config["table_observed"], config["table_footprint"])
                    self.assertEqual(source["execution_scope"], "operator_cleared_table_workspace")
                    self.assertEqual(manifest["execution_scope"], "operator_cleared_table_workspace")
                else:
                    self.assertNotEqual(config["table_observed"], config["table_footprint"])
                    self.assertEqual(source["execution_scope"], "observed_table_workspace")
                loaded_config = Config(**config)
                # This tabletop strip is outside the synthetic stereo domain.
                if cleared:
                    check_observed_box(loaded_config, [0.01, 0.3, 1.1], [0.03, 0.32, 1.15])
                else:
                    with self.assertRaisesRegex(ValueError, "unobserved_table_region"):
                        check_observed_box(loaded_config, [0.01, 0.3, 1.1], [0.03, 0.32, 1.15])
                # Operator clearance does not allow the held cube through the table.
                self.assertTrue(cube_penetrates(loaded_config, [0.4, 0.3, 0.99]))

    def test_operator_scene_does_not_require_camera_coverage_calculation(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("astrabot.robot.scene_builder.observed_table", side_effect=ValueError("no_common_table_view")),
        ):
            self.build_scene(Path(directory), True)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("astrabot.robot.scene_builder.observed_table", side_effect=ValueError("no_common_table_view")),
            self.assertRaisesRegex(ValueError, "no_common_table_view"),
        ):
            self.build_scene(Path(directory), False)

    def test_missing_A_depth_blocks_even_with_operator_clearance(self):
        with tempfile.TemporaryDirectory() as directory:
            builder, measured = self.make_builder(Path(directory), True)
            del measured["letters"]["A"]
            annotation = annotation_from_decision(
                decision(), builder.candidates, builder.left.shape, DIMENSIONS, operator_cleared_workspace=True
            )
            with patch("astrabot.robot.scene_builder.StereoTracker") as tracker:
                tracker.return_value.locate_images.return_value = measured
                with self.assertRaisesRegex(ValueError, "A 必须有可靠双目深度"):
                    builder.build(annotation)

    async def test_exported_operator_scene_observes_A_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = self.build_scene(root, True)
            letters = json.loads((root / "scene-letters.json").read_text())["letters"]
            image_path = root / "scene-review.jpg"
            model = Mock()
            model.poses.return_value = {name: np.eye(4) for name in ("left_index_distal", "left_thumb_distal")}
            with patch.object(TrialObserver, "_glyph", return_value=np.ones((5, 5), np.uint8)):
                observer = TrialObserver(None, model, builder.config, letters, image_path, root / "observations")
            tracker = Mock()
            tracker.locate_images.return_value = {"letters": copy.deepcopy(letters), "errors": {}}

            def capture(_config, output, _client):
                output.mkdir(parents=True)
                return image_path, image_path, {"body_after": {"body": [], "hands": []}, "acquired_monotonic": 1}

            with (
                patch("astrabot.robot.trial_observer.capture", side_effect=capture),
                patch("astrabot.robot.trial_observer.load", return_value=(None, None, None)),
                patch("astrabot.robot.trial_observer.StereoTracker", return_value=tracker),
                patch.object(observer, "project", return_value=np.array([30, 30])),
                patch.object(observer, "occluded", return_value=False),
                patch.object(observer, "reacquire_pixels", return_value=({"A": [30, 30]}, None)),
                patch.object(observer, "_glyph", return_value=observer.template),
            ):
                result = await observer("before_approach")
            self.assertEqual(set(tracker.locate_images.call_args.args[2]), {"A"})
            self.assertEqual(result["scene_clear_source"], "operator_cleared")
            self.assertIn("not visually checked", result["scope"])


if __name__ == "__main__":
    unittest.main()
