"""Offline scene visibility and obstacle checks; no API or robot connections."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.config import file_digest
from astrabot.robot.scene_builder import SceneBuilder, project, table_corners
from astrabot.robot.scene_identification import SceneVision, annotation_from_decision
from astrabot.robot.scene_review import automatic_scene
from astrabot.robot.table_edges import fit_table_edges


def valid_decision(fully_visible=False):
    """Return visible straight segments without inventing a sharp table corner."""
    return {
        "target_letter": "A",
        "target_id": 30,
        "cube_ids": list(range(9, 35)),
        "table_edge_mode": "adjacent_straight_edges",
        "near_edge_normalized": [[200, 730], [500, 740], [800, 750]],
        "left_edge_normalized": [[100, 700], [170, 550], [240, 400]],
        "target_confidence": 0.99,
        "selection_confidence": 0.97,
        "table_edge_confidence": 0.95,
        "unique_target": True,
        "all_cube_glyphs_numbered": True,
        "all_cubes_upright": True,
        "selection_complete": True,
        "table_edge_visible": True,
        "tabletop_fully_visible": fully_visible,
        "tabletop_clear_except_cubes": True,
        "uncertainty_reasons": [],
        "evidence": "All 26 visible cubes are numbered; the near and left straight edges are clear, with rounded corners.",
    }


CANDIDATES = [{"id": i} for i in range(1, 35)]
DIMENSIONS = {"table_width_mm": 1200, "table_depth_mm": 600, "table_thickness_mm": 10}
SHAPE = (1200, 1920, 3)


class SceneIdentificationTests(unittest.TestCase):
    def test_requested_first_letter_is_not_hardcoded_to_a(self):
        decision = valid_decision()
        decision["target_letter"] = "C"
        annotation = annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS, target_letter="C")
        self.assertEqual(annotation["target_id"], 30)
        with self.assertRaisesRegex(ValueError, "scene_target_must_be_A"):
            annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)
        with tempfile.TemporaryDirectory() as directory:
            with patch("astrabot.vision.OfficialOpenAI"):
                vision = SceneVision(Path(directory), timeout=1, glyph_mode="real")
            with (
                patch("astrabot.robot.scene_identification.candidate_images", return_value=[]),
                patch.object(vision, "_call", return_value=decision) as call,
            ):
                vision.identify(SimpleNamespace(target_letter="C", operator_cleared_workspace=True))
            self.assertIn('"target_letter":"C"', call.call_args.args[1])
            self.assertNotIn('"target_letter":"A"', call.call_args.args[1])

    def test_raised_robot_notes_allow_annotation_without_certifying_hidden_table(self):
        decision = valid_decision()
        decision.update(
            robot_self_candidate_ids=[1, 2, 3, 4],
            unresolved_cube_ids=[],
            visibility_notes=["Recognizable raised forearms; visible samples lie either side of an edge gap."],
            table_edge_confidence=0.88,
        )
        annotation = annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)
        self.assertEqual(annotation["cube_ids"], decision["cube_ids"])
        self.assertFalse(decision["tabletop_fully_visible"])
        self.assertEqual(len(annotation["near_edge_px"]), 3)

    def test_lower_confidence_only_applies_to_two_line_geometry(self):
        for key, score, mode, accepted in (
            ("table_edge_confidence", 0.85, "adjacent_straight_edges", True),
            ("table_edge_confidence", 0.849, "adjacent_straight_edges", False),
            ("table_edge_confidence", 0.88, "visible_corner", False),
            ("target_confidence", 0.88, "adjacent_straight_edges", False),
            ("selection_confidence", 0.88, "adjacent_straight_edges", False),
        ):
            with self.subTest(key=key, score=score, mode=mode):
                decision = valid_decision()
                decision[key] = score
                decision["table_edge_mode"] = mode
                if accepted:
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)
                else:
                    with self.assertRaisesRegex(ValueError, f"scene_identification_uncertain:{key}"):
                        annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_robot_ids_cannot_hide_or_replace_selected_cubes(self):
        for robot_ids in ([30], [999], [True], [1, 1], "1"):
            with self.subTest(robot_ids=robot_ids):
                decision = valid_decision()
                decision["robot_self_candidate_ids"] = robot_ids
                with self.assertRaisesRegex(ValueError, "scene_(invalid_robot_self_candidate_ids|robot_self_selected)"):
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_unresolved_cube_is_blocking_even_if_booleans_claim_success(self):
        for cube_id in (16, 30, 1):
            with self.subTest(cube_id=cube_id):
                decision = valid_decision()
                decision["unresolved_cube_ids"] = [cube_id]
                decision["visibility_notes"] = ["Robot visible."]
                with self.assertRaisesRegex(ValueError, f"unresolved_cube_ids={cube_id}"):
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_robot_notes_do_not_override_unknown_objects_or_occluded_workspace(self):
        for reason in ("Cable crosses the tabletop.", "Unidentified dark object.", "Workspace hidden behind hand."):
            with self.subTest(reason=reason):
                decision = valid_decision()
                decision.update(robot_self_candidate_ids=[1], visibility_notes=["Raised hand visible."])
                decision["uncertainty_reasons"] = [reason]
                with self.assertRaisesRegex(ValueError, "scene_identification_uncertain"):
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_recorded_M_uncertainty_is_not_silently_cleared(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_raised_scene_identification.json").read_text())
        with self.assertRaises(ValueError) as raised:
            annotation_from_decision(fixture["decision"], fixture["candidates"], SHAPE, DIMENSIONS)
        message = str(raised.exception)
        self.assertNotIn("table_edge_confidence=0.88", message)
        self.assertIn("Cube 16 (M)", message)
        self.assertIn("all_cubes_upright", message)

    def test_prompt_distinguishes_robot_presence_from_missing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("astrabot.vision.OfficialOpenAI"):
                vision = SceneVision(Path(directory), timeout=1, glyph_mode="real")
            with (
                patch("astrabot.robot.scene_identification.candidate_images", return_value=[]),
                patch.object(vision, "_call", return_value=valid_decision()) as call,
            ):
                vision.identify(SimpleNamespace())
            prompt = call.call_args.args[1]
            for text in (
                "robot_self_candidate_ids",
                "unresolved_cube_ids",
                "visibility_notes",
                "Never place a sample inside an occluded gap",
                "Cables are not covered by this exception",
                "Hidden space is not certified clear",
            ):
                self.assertIn(text, prompt)

    def test_both_modes_request_visible_edge_points_outside_forearms(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("astrabot.vision.OfficialOpenAI"):
                vision = SceneVision(Path(directory), timeout=1, glyph_mode="real")
            for cleared in (False, True):
                with (
                    self.subTest(cleared=cleared),
                    patch("astrabot.robot.scene_identification.candidate_images", return_value=[]),
                    patch.object(vision, "_call", return_value={}) as call,
                ):
                    vision.identify(SimpleNamespace(operator_cleared_workspace=cleared))
                    prompt = call.call_args.args[1]
                    for instruction in (
                        "OUTSIDE BOTH",
                        "LEFT of the left forearm",
                        "nearest reliable STRAIGHT point",
                        "Never sample through the occluder",
                        "300 mm",
                    ):
                        self.assertIn(instruction, prompt)

    def test_accepted_visual_confidence_still_requires_metric_geometry_before_export(self):
        decision = valid_decision()
        decision["table_edge_confidence"] = 0.88
        line = np.array([[0.05, 0, 1], [0.3, 0, 1], [0.55, 0, 1]])
        with tempfile.TemporaryDirectory() as directory:
            builder = SimpleNamespace(
                candidates=CANDIDATES,
                output=Path(directory),
                left=SimpleNamespace(shape=SHAPE),
                build=Mock(
                    side_effect=lambda *args, **kwargs: fit_table_edges(
                        line, line + [0, 0.2, 0], np.array([0, 0, 1]), 1.2, 0.6
                    )
                ),
                export_automatic=Mock(),
            )
            with patch("astrabot.robot.scene_identification.SceneVision") as vision:
                vision.return_value.identify.return_value = decision
                with self.assertRaisesRegex(ValueError, "table_edges_not_perpendicular"):
                    automatic_scene(builder, DIMENSIONS, 120)
            builder.build.assert_called_once()
            builder.export_automatic.assert_not_called()

    def test_cropped_table_allows_pixel_annotation(self):
        annotation = annotation_from_decision(valid_decision(), CANDIDATES, SHAPE, DIMENSIONS)
        self.assertEqual(annotation["target_id"], 30)
        self.assertEqual(len(annotation["cube_ids"]), 26)
        self.assertEqual(annotation["table_edge_mode"], "adjacent_straight_edges")
        np.testing.assert_allclose(annotation["near_edge_px"][0], [200 * 1919 / 1000, 730 * 1199 / 1000])
        np.testing.assert_allclose(annotation["left_edge_px"][-1], [240 * 1919 / 1000, 400 * 1199 / 1000])

    def test_missing_or_invalid_side_samples_do_not_fall_back_to_corner(self):
        for points in (None, [], [[100, 700], [170, 550]], [[100, 700], [170, 550], [True, 400]]):
            with self.subTest(points=points):
                decision = valid_decision()
                decision["left_edge_normalized"] = points
                with self.assertRaisesRegex(ValueError, "scene_requires_3_to_8_left_edge_points"):
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_legacy_corner_decisions_remain_readable(self):
        decision = valid_decision()
        del decision["table_edge_mode"], decision["left_edge_normalized"]
        decision["near_edge_normalized"] = [[114, 741], [938, 793]]
        annotation = annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)
        self.assertNotIn("table_edge_mode", annotation)
        self.assertEqual(len(annotation["near_edge_px"]), 2)

    def test_low_confidence_includes_explanation_and_other_blockers(self):
        decision = valid_decision()
        decision.update(table_edge_confidence=0.55, table_edge_visible=False)
        decision["uncertainty_reasons"] = ["The left straight edge is occluded."]
        with self.assertRaises(ValueError) as raised:
            annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)
        for detail in ("table_edge_confidence=0.55", "table_edge_visible", "left straight edge is occluded"):
            self.assertIn(detail, str(raised.exception))

    def test_visibility_must_still_be_explicit_boolean(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value):
                decision = valid_decision(value)
                with self.assertRaisesRegex(ValueError, "tabletop_fully_visible_must_be_boolean"):
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_other_required_evidence_still_blocks(self):
        for key in (
            "unique_target",
            "all_cube_glyphs_numbered",
            "all_cubes_upright",
            "selection_complete",
            "table_edge_visible",
            "tabletop_clear_except_cubes",
        ):
            with self.subTest(key=key):
                decision = valid_decision()
                decision[key] = False
                with self.assertRaisesRegex(ValueError, f"scene_identification_incomplete:{key}"):
                    annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_uncertain_corner_still_blocks(self):
        decision = valid_decision()
        decision["uncertainty_reasons"] = ["The rounded near-left corner is ambiguous."]
        with self.assertRaisesRegex(ValueError, "scene_identification_uncertain:.*rounded"):
            annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)

    def test_all_blockers_are_reported_together(self):
        decision = valid_decision()
        decision.update(tabletop_clear_except_cubes=False, table_edge_visible=False)
        decision["uncertainty_reasons"] = ["Black remote on the tabletop.", "Rounded near-left corner is ambiguous."]
        with self.assertRaises(ValueError) as raised:
            annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS)
        message = str(raised.exception)
        for detail in ("table_edge_visible", "tabletop_clear_except_cubes", "Black remote", "Rounded", "重新拍照"):
            self.assertIn(detail, message)
        self.assertNotIn("incomplete:tabletop_fully_visible", message)

    def test_table_dimensions_reconstruct_off_image_corners(self):
        geometry = SimpleNamespace(K_left=np.diag([1000.0, 1000.0, 1.0]), D_left=np.zeros(5))
        transform = np.eye(4)
        letters = {
            str(i): {"estimated_xyz": [x, y, 1.02]}
            for i, (x, y) in enumerate(((0.2, 0.2), (0.4, 0.2), (0.2, 0.4), (0.4, 0.4)))
        }
        corners = table_corners([[0, 0], [200, 0]], np.array([0, 0, 1.0]), geometry, transform, letters, 1.2, 0.6)
        np.testing.assert_allclose(corners, [[0, 0, 1], [1.2, 0, 1], [1.2, 0.6, 1], [0, 0.6, 1]])
        self.assertGreater(project(corners, geometry, transform)[:, 0].max(), 640)

    def test_recorded_failure_stops_before_stereo_and_export(self):
        decision = valid_decision()
        decision["tabletop_clear_except_cubes"] = False
        decision["uncertainty_reasons"] = [
            "The tabletop extends beyond the right image boundary.",
            "A black remote-control-like object rests on the upper-right tabletop.",
            "The near-left tabletop corner is rounded, making its exact endpoint slightly ambiguous.",
        ]
        with tempfile.TemporaryDirectory() as directory:
            builder = SimpleNamespace(
                candidates=CANDIDATES,
                output=Path(directory),
                left=SimpleNamespace(shape=SHAPE),
                build=Mock(),
                export_automatic=Mock(),
            )
            with patch("astrabot.robot.scene_identification.SceneVision") as vision:
                vision.return_value.identify.return_value = decision
                with self.assertRaisesRegex(ValueError, "tabletop_clear_except_cubes"):
                    automatic_scene(builder, DIMENSIONS, 120)
                vision.return_value.identify.assert_called_once_with(builder)
            builder.build.assert_not_called()
            builder.export_automatic.assert_not_called()
            self.assertEqual(json.loads((builder.output / "identification.json").read_text()), decision)


class SceneVisibilityExportTests(unittest.TestCase):
    def test_export_preserves_visibility_and_completeness(self):
        for fully_visible in (False, True):
            with self.subTest(fully_visible=fully_visible), tempfile.TemporaryDirectory() as directory:
                # Supply an already-built geometry draft; exercise the real evidence
                # validation and file export without calibration files or hardware.
                builder = SceneBuilder.__new__(SceneBuilder)
                builder.output = Path(directory)
                builder.capture_dir = builder.output / "capture"
                builder.candidates = CANDIDATES
                builder.left = SimpleNamespace(shape=SHAPE)
                builder.exported = False
                builder.verify_sources = Mock()
                decision = valid_decision(fully_visible)
                decision.update(
                    table_edge_confidence=0.88,
                    robot_self_candidate_ids=[1, 2],
                    visibility_notes=["Raised robot parts excluded from cube candidates."],
                    unresolved_cube_ids=[],
                )
                path = builder.output / "identification.json"
                path.write_text(json.dumps(decision))
                builder.draft = {
                    "identification": {"path": str(path), "sha256": file_digest(path)},
                    "annotation": annotation_from_decision(decision, CANDIDATES, SHAPE, DIMENSIONS),
                    "input": {"scene_complete": False},
                    "config": {"scene_measured": True, "tcp_measured": False},
                    "letters": {},
                }
                draft_path = builder.output / "draft.json"
                draft_path.write_text(json.dumps(builder.draft))
                builder.draft_hash = file_digest(draft_path)

                result = builder.export_automatic()

                source = json.loads((builder.output / "scene-input.json").read_text())
                review = json.loads((builder.output / "scene-review.json").read_text())
                manifest = json.loads((builder.output / "export.json").read_text())
                self.assertIs(source["scene_complete"], fully_visible)
                self.assertIs(result["scene_complete"], fully_visible)
                self.assertIs(manifest["scene_complete"], fully_visible)
                self.assertIs(review["review"]["tabletop_fully_visible"], fully_visible)
                self.assertFalse(review["review"]["operator_confirmed"])
                self.assertFalse(manifest["hardware_ready"])
                self.assertTrue(builder.exported)


if __name__ == "__main__":
    unittest.main()
