"""One vision request for scene identity and table pixels; no robot interface."""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..vision import AstraVision

# Compatibility for the two metric-only advisories emitted before measurement_notes
# was introduced. Match whole statements, never keywords: an unseen/wrong edge or
# an actual height concern must not be discarded as a generic measurement caveat.
LEGACY_MEASUREMENT_NOTES = frozenset(
    {
        "Edge coordinates are visual estimates; metric span, perpendicularity, residuals and extrapolation require local validation.",
        "The nominal 40 mm cube dimensions and precise resting heights cannot be independently measured from this single image.",
    }
)

TABLE_EDGE_SAMPLING_GUIDANCE = """
IMPORTANT EDGE-SAMPLE COVERAGE:
Inspect the ENTIRE original image for the near edge before selecting samples, including OUTSIDE BOTH
forearms. Do not restrict near-edge samples to the opening BETWEEN the arms. A near-edge strip to the
LEFT of the left forearm is especially useful and must not be omitted when clearly visible.
For each edge, select its nearest reliable STRAIGHT point to the near-left rounded transition, outside
the curved arc. If a straight near-edge strip is visible between that arc and the left forearm, include
one or two well-separated points on that strip, then distribute the remaining points across other visible
portions of the same edge. Apply the same nearest-visible-point rule to the left edge, ordered near to far.
Use 3 to 8 points per edge in total. Robot occlusion may separate sample groups; that does not justify
discarding a visible group. Never sample through the occluder, on a cable, along the lower thickness edge,
or on the rounded arc. Do not invent a corner point. Local fitting only permits extrapolation from the
nearest sample to the reference corner up to 300 mm (or half that edge's measured length if smaller).
Choose actual visible near-corner straight portions to reduce extrapolation, not fabricated coordinates.
In evidence, briefly state which separated edge portions were used and why any visible near-corner
portion was excluded. If no reliable nearer portion exists, report that fact instead of making one up.
"""


def validate_dimensions(width: float, depth: float, thickness: float) -> dict:
    """Validate measured table dimensions before making a paid request."""
    result = {}
    for name, value, lower, upper in (
        ("table_width_mm", width, 300, 2500),
        ("table_depth_mm", depth, 200, 1500),
        ("table_thickness_mm", thickness, 1, 100),
    ):
        if type(value) not in (int, float) or not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"invalid_{name}:{lower}..{upper}")
        result[name] = value
    return result


def annotation_from_decision(
    decision: dict,
    candidates: list,
    shape: tuple,
    dimensions: dict,
    *,
    operator_cleared_workspace=False,
    target_letter="A",
) -> dict:
    """Validate visual evidence and convert normalized pixels to original pixels."""
    if not isinstance(decision, dict):
        raise TypeError("scene_decision_must_be_object")
    mode = decision.get("table_edge_mode", "visible_corner")
    known = {item["id"] for item in candidates}
    classified = {}
    for key in ("robot_self_candidate_ids", "unresolved_cube_ids"):
        values = decision.get(key, [])
        if (
            not isinstance(values, list)
            or any(type(value) is not int or value not in known for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"scene_invalid_{key}")
        classified[key] = set(values)
    notes = decision.get("visibility_notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) or not note.strip() for note in notes):
        raise ValueError("scene_invalid_visibility_notes")
    confidence_errors = []
    confidence_keys = (
        ("target_confidence", "table_edge_confidence")
        if operator_cleared_workspace
        else ("target_confidence", "selection_confidence", "table_edge_confidence")
    )
    for key in confidence_keys:
        value = decision.get(key)
        # Two sampled lines must still pass the independent metric fit before
        # export. A legacy corner estimate has no equivalent fit validation.
        minimum = 0.85 if key == "table_edge_confidence" and mode == "adjacent_straight_edges" else 0.9
        if type(value) not in (int, float) or not np.isfinite(value) or not minimum <= value <= 1:
            confidence_errors.append(f"scene_identification_uncertain:{key}={value}")
    if decision.get("target_letter") != target_letter:
        raise ValueError(f"scene_target_must_be_{target_letter}")
    required = (
        "unique_target",
        "all_cube_glyphs_numbered",
        "all_cubes_upright",
        "selection_complete",
        "table_edge_visible",
        "tabletop_clear_except_cubes",
    )
    if operator_cleared_workspace:
        required = ("target_verified", "target_upright", "table_edge_visible")
    failed = [key for key in required if decision.get(key) is not True]
    unresolved = classified["unresolved_cube_ids"]
    if operator_cleared_workspace:
        unresolved = unresolved.intersection([decision.get("target_id")])
    if unresolved:
        failed.append("unresolved_cube_ids=" + ",".join(map(str, sorted(unresolved))))
    # Dimensions can reconstruct an off-image corner, but not unseen obstacles.
    # Keep visibility explicit so export cannot claim whole-table completeness.
    if type(decision.get("tabletop_fully_visible")) is not bool:
        failed.append("tabletop_fully_visible_must_be_boolean")
    reasons = decision.get("uncertainty_reasons")
    measurement_notes = []
    if operator_cleared_workspace:
        # Only the explicit caller selects this policy; never trust a vision
        # response to grant itself permission to ignore non-target obstacles.
        reasons = []
        measurement_notes = decision.get("measurement_notes", [])
        if not isinstance(measurement_notes, list) or any(
            not isinstance(v, str) or not v.strip() for v in measurement_notes
        ):
            failed.append("invalid_measurement_notes")
            measurement_notes = []
        else:
            measurement_notes = list(measurement_notes)
        for key in ("target_uncertainty_reasons", "table_uncertainty_reasons"):
            entries = decision.get(key)
            if not isinstance(entries, list) or any(not isinstance(v, str) or not v.strip() for v in entries):
                failed.append(f"invalid_{key}")
            else:
                for entry in entries:
                    if (
                        key == "table_uncertainty_reasons"
                        and mode == "adjacent_straight_edges"
                        and entry.strip() in LEGACY_MEASUREMENT_NOTES
                    ):
                        measurement_notes.append(entry)
                    else:
                        reasons.append(entry)
    if confidence_errors or failed or not isinstance(reasons, list) or reasons:
        details = list(confidence_errors)
        if failed:
            details.append(f"scene_identification_incomplete:{','.join(failed)}")
        if not isinstance(reasons, list) or reasons:
            details.append(f"scene_identification_uncertain:{json.dumps(reasons, ensure_ascii=False)}")
        if "tabletop_clear_except_cubes" in failed:
            details.append(
                "可见桌面杂物或线缆占用尚未排除；请按 evidence 定位，确认或移除后重新拍照。机器人手臂本身不属于桌面杂物。"
            )
        if operator_cleared_workspace and (unresolved or "target_upright" in failed):
            details.append("选定 A 的位置、朝向或独立摆放无法确认；改善目标遮挡后重新拍照。")
        elif "all_cubes_upright" in failed or unresolved:
            details.append("部分积木的位置、朝向或独立摆放无法确认；改善遮挡后重新拍照，不会忽略这些积木。")
        if "table_edge_visible" in failed:
            details.append("自动模式需要近边与左边可见的直线段采样点；圆角无需提供尖角坐标。")
        details.append(f"evidence:{decision.get('evidence', '')}")
        raise ValueError("\n".join(details))
    if not isinstance(decision.get("evidence"), str) or not decision["evidence"].strip():
        raise ValueError("scene_identification_missing_evidence")
    ids, target = decision.get("cube_ids"), decision.get("target_id")
    if not isinstance(ids, list) or not 4 <= len(ids) <= 64 or any(type(i) is not int for i in ids):
        raise ValueError("scene_requires_4_to_64_numbered_cubes")
    if len(ids) != len(set(ids)) or any(i not in known for i in ids):
        raise ValueError("scene_duplicate_or_unknown_candidate")
    if classified["robot_self_candidate_ids"].intersection(ids):
        raise ValueError("scene_robot_self_selected_as_cube")
    if type(target) is not int or target not in ids:
        raise ValueError("scene_A_not_in_selected_cubes")
    if mode not in ("visible_corner", "adjacent_straight_edges"):
        raise ValueError("scene_unknown_table_edge_mode")
    if mode == "visible_corner" and "left_edge_normalized" in decision:
        raise ValueError("scene_left_edge_requires_adjacent_straight_edges_mode")
    names = ("near", "left") if mode == "adjacent_straight_edges" else ("near",)
    count_min, count_max = (3, 8) if len(names) == 2 else (2, 2)
    height, width = shape[:2]
    edges = {}
    for name in names:
        edge = decision.get(f"{name}_edge_normalized")
        if (
            not isinstance(edge, list)
            or not count_min <= len(edge) <= count_max
            or any(not isinstance(point, list) or len(point) != 2 for point in edge)
            or any(type(value) not in (int, float) for point in edge for value in point)
        ):
            raise ValueError(f"scene_requires_{count_min}_to_{count_max}_{name}_edge_points")
        edge = np.asarray(edge, float)
        if not np.isfinite(edge).all() or np.any(edge < 0) or np.any(edge > 1000):
            raise ValueError("scene_edge_coordinates_outside_0_to_1000")
        edges[f"{name}_edge_px"] = (edge * [width - 1, height - 1] / 1000).tolist()
    if mode == "adjacent_straight_edges":
        edges["table_edge_mode"] = mode
    return {
        "cube_ids": sorted(ids),
        "target_id": target,
        **edges,
        **({"workspace_policy": "operator_cleared"} if operator_cleared_workspace else {}),
        **({"measurement_notes": measurement_notes} if measurement_notes else {}),
        **validate_dimensions(
            dimensions["table_width_mm"], dimensions["table_depth_mm"], dimensions["table_thickness_mm"]
        ),
    }


def operator_cleared_prompt() -> str:
    """Request target and table measurements for an explicitly operator-cleared experiment."""
    return """Measure A and the table for a supervised robot experiment.
The OPERATOR, not vision, is responsible for keeping non-target objects out of the arm and grasp path.
This mode does NOT verify surrounding obstacles, a complete cube inventory, or whole-table clearance.
Image 1 is the full original image, image 2 the same image with numbered components. Remaining images
are enlarged glyph crops of those SAME candidate IDs, not spatial layouts.
Choose ONE clearly identifiable visible A as the target. Do not require proof that no other A exists;
hidden non-target glyphs are irrelevant, and multiple visible As are allowed if one target ID is selected.
Set target_verified true only for that selected readable A. It must be an upright, resting 40 mm cube with
a visible glyph location usable for stereo. Report uncertainty about A itself; never assume A's pose.
Select A and preferably 8 to 12 total clear visible 40 mm cube top glyph locations as TABLE MEASUREMENT
SAMPLES only (at least four if fewer are reliable). Use redundant samples from NEAR, MIDDLE and FAR rows,
spread across the table's LEFT and RIGHT sides. When visible, select at least two samples in each of
the near and far depth bands: after one bad depth is rejected the remainder must still span both table
directions. Do not choose mainly one row plus a single distant point, or only the gap between raised arms.
Inspect the visible far row behind the fingers and the strips outside both forearms. Prefer clear,
upright, unstacked samples with an intact glyph on the TOP face. Choose only ONE candidate per physical
cube; fragmented glyph components are not independent measurements. Do not hard-code IDs from old images.
You need not read non-target letters or enumerate all cubes. Skip obscured or ambiguous non-target samples.
Never add a doubtful sample merely to meet the preferred count. Local stereo and plane fitting validate
these samples; they will NOT become obstacles or be tracked during grasping. Robot fingers are not samples.
Non-target occlusion, missing non-target letters, other objects, robot parts and cables do not block
this mode; put those observations in visibility_notes. Do not certify those areas or objects as clear.
The table is rectangular, optionally with rounded corners; its measured dimensions are supplied locally.
Select 3 to 8 ordered visible points on EACH of the near and left straight physical tabletop edges.
near_edge_normalized runs image-left to image-right; left_edge_normalized runs near to far.
Separated visible portions on the SAME line are allowed. Never sample an occluded gap, arm, shadow,
rounded arc, far edge or lower thickness edge. No physical sharp corner is required. Local fitting still
checks span >=150 mm per edge, perpendicularity, residuals and bounded extrapolation. Report actual VISUAL
ambiguity (e.g. confusing the top edge with the lower thickness edge, or no visible straight segment) in
table_uncertainty_reasons. Do not invent points to pass.
Separate visual failures from measurements pending local validation:
- target_uncertainty_reasons: actual ambiguity of the selected A's identity or visible resting pose only.
- table_uncertainty_reasons: actual ambiguity of the selected physical tabletop edges only.
- measurement_notes: informational metric caveats, NEVER a visual failure. The 40 mm cube size is a supplied
  prior, not something to prove from one image. Local stereo checks A's height (40 mm +/-8 mm), plane fit,
  edge span, perpendicularity, residuals and extrapolation AFTER your response. Do not put inability to
  measure these quantities from a single image in either uncertainty list or lower visual confidence for it.
For clear A and clear sampled edges, both uncertainty lists must be empty even though local metric checks
are still pending. Real visual ambiguity must never be hidden in measurement_notes or visibility_notes.
All edge coordinates are [x,y] normalized to 0..1000 over the ORIGINAL full image, not the crops.
Return JSON:
{"target_letter":"A","target_id":1,"cube_ids":[1,2,3,4],
 "target_confidence":0.0,"target_verified":false,"target_upright":false,
 "table_edge_mode":"adjacent_straight_edges","table_edge_confidence":0.0,"table_edge_visible":false,
 "near_edge_normalized":[[200,800],[500,820],[800,840]],
 "left_edge_normalized":[[160,750],[240,600],[320,450]],
 "tabletop_fully_visible":false,"robot_self_candidate_ids":[],"unresolved_cube_ids":[],
 "target_uncertainty_reasons":[],"table_uncertainty_reasons":[],"measurement_notes":[],"visibility_notes":[],
 "evidence":"Describe A and the measured table evidence; do not claim surrounding clearance."}
Confidence is 0..1. Missing or ambiguous A, uncertain A pose and unreliable table edges remain blocking.
"""


def candidate_images(builder, directory: Path) -> list[Path]:
    """Render readable, consistently numbered scene and crop pages for one call."""
    original = Image.fromarray(builder.left[:, :, ::-1])
    marked = original.copy()
    marked.thumbnail((960, 540))
    scale_x, scale_y = marked.width / original.width, marked.height / original.height
    draw = ImageDraw.Draw(marked)
    for item in builder.candidates:
        x, y, width, height = item["bbox"]
        draw.rectangle(
            (x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y),
            outline="red",
            width=1,
        )
        label_x, label_y = x * scale_x, max(0, y * scale_y - 12)
        draw.text((label_x, label_y), str(item["id"]), fill="red", stroke_width=1, stroke_fill="white")
    marked_path = directory / "numbered-scene.jpg"
    marked.save(marked_path)
    images = [builder.capture_dir / "left.jpg", marked_path]
    # Each page stays below the common client's thumbnail limit; glyphs remain legible.
    for start in range(0, len(builder.candidates), 32):
        page = builder.candidates[start : start + 32]
        sheet = Image.new("RGB", (880, ((len(page) + 7) // 8) * 110), "white")
        sheet_draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(page):
            x, y, width, height = item["bbox"]
            crop = original.crop(
                (
                    max(0, x - 12),
                    max(0, y - 12),
                    min(original.width, x + width + 12),
                    min(original.height, y + height + 12),
                )
            )
            crop.thumbnail((90, 82))
            gx, gy = index % 8 * 110, index // 8 * 110
            sheet.paste(crop, (gx + 10, gy + 24))
            sheet_draw.text((gx + 8, gy + 5), str(item["id"]), fill="black")
        path = directory / f"glyphs-{start // 32 + 1}.jpg"
        sheet.save(path)
        images.append(path)
    (directory / "candidates.json").write_text(json.dumps(builder.candidates, indent=2) + "\n")
    return images


class SceneVision(AstraVision):
    """Reuse the existing API client and request logs without automatic retries."""

    def identify(self, builder) -> dict:
        """Identify A, cube candidates, and two adjacent straight table edges once."""
        images = candidate_images(builder, self.directory)
        prompt = """Prepare a real robot's table scene from this recorded image.
Image 1 is the original full camera image. Image 2 is the SAME full image with numbered blue-ink
components. Remaining images are enlarged crops of the SAME candidate IDs, NOT spatial table layouts.
Use the crops to read letters and the full image to assess objects, visibility and table edges.
Select ALL visible cubes on the main work table, including non-target letters, one glyph candidate per cube.
Exclude background components. Identify exactly one letter A among them. All cubes are known 40 mm
cubes; upright means letter face up, individually resting on the same table, not stacked or tilted.
The robot may already be holding its arms in the raised ready pose. Clearly recognizable robot grippers,
fingers and forearms are robot self-geometry, NOT foreign tabletop clutter or letter cubes. List their
numbered image components in robot_self_candidate_ids and exclude them from cube_ids. Robot presence
alone does not make tabletop_clear_except_cubes false. Never classify an ambiguous dark object as robot
self merely to pass the check. Cables are not covered by this exception: a cable resting on or crossing
the tabletop, or uncertain cable contact/occupation of the workspace, remains a blocking reason.
Do not require a completely visible silhouette for every cube. Partial outline occlusion is acceptable
ONLY when the visible glyph location and top/side evidence still establish its identity, position and
upright, independent resting state. Include that cube as an obstacle; do not drop it from cube_ids.
If this evidence is insufficient, list its candidate ID in unresolved_cube_ids, set the relevant required
booleans false and explain exactly what is hidden. A needs a reliable visible glyph location; its depth
is checked independently by stereo. Never infer a hidden cube's pose from an earlier photo or this task.
Do not assume every cube was detected: inspect the original image for unnumbered cubes and other objects.
The current scene format supports these cubes and a rectangular table with optionally rounded corners.
It does not support circular tables. Report other tabletop
objects, missing candidates, duplicate A, or uncertain required landmarks instead of ignoring them.
Measured table width, depth and thickness are supplied separately to local geometry. The far/right corners
need not be in the image: the visible NEAR and LEFT straight edges and measured dimensions define
the rectangle. Set tabletop_fully_visible truthfully; false alone is not an uncertainty reason or a reason
to mark selection_complete false. It limits the export to an incomplete scene for offline planning.
selection_complete and tabletop_clear_except_cubes describe the VISIBLE main tabletop only, never unseen
areas. Robot self-presence and resolved partial outlines belong in visibility_notes (informational).
uncertainty_reasons contains BLOCKING uncertainties only: unknown objects, unresolved cube geometry,
missing cube candidates, ambiguous landmarks or unresolved occlusion of the tabletop workspace.
Do not move a real uncertainty into visibility_notes to get a pass. Hidden space is not certified clear.
Select 3 to 8 well-spaced visible points on EACH of two adjacent physical tabletop edges:
near_edge_normalized: the STRAIGHT near edge, ordered from image-left toward image-right;
left_edge_normalized: the STRAIGHT left side edge, ordered from near toward far.
Use points distributed over a long straight portion, including a point reasonably close to the near-left
rounded transition but OUTSIDE its curved arc. Do not sample rounded arcs, paper edges, shadows, the far
edge, or the lower edge of the table thickness. All sampled points must be visible in the original image.
Do NOT identify or output a corner endpoint or invent a virtual intersection. Local calibrated geometry
fits the two straight lines and computes their intersection as a reference corner of an enclosing rectangle.
A rounded transition with no physical sharp endpoint is EXPECTED and is not itself an uncertainty reason.
table_edge_visible and table_edge_confidence refer to locating BOTH STRAIGHT LINES, not a sharp corner
or an uninterrupted edge. Separate visible portions of the SAME physical line can supply the samples;
gaps behind a forearm alone are not a failure. Never place a sample inside an occluded gap or on the arm.
Report uncertainty if the remaining visible portions are too short, ambiguous, or on different boundaries.
Local metric checks still require at least 150 mm sample span per edge, near-perpendicular lines and
bounded fit residuals/extrapolation. Score confidence in the chosen visible points, not edge continuity.
Coordinates are [x,y] normalized to 0..1000 across the ORIGINAL FULL image:
top-left [0,0], bottom-right [1000,1000]. Never output crop coordinates or estimated world coordinates.
Return JSON with these keys (values below illustrate types only, they are NOT answers):
{"target_letter":"A","target_id":1,"cube_ids":[1,2,3,4],
 "table_edge_mode":"adjacent_straight_edges",
 "near_edge_normalized":[[200,800],[500,820],[800,840]],
 "left_edge_normalized":[[160,750],[240,600],[320,450]],
 "target_confidence":0.0,"selection_confidence":0.0,"table_edge_confidence":0.0,
 "unique_target":false,"all_cube_glyphs_numbered":false,"all_cubes_upright":false,
 "selection_complete":false,"table_edge_visible":false,"tabletop_fully_visible":false,
 "tabletop_clear_except_cubes":false,"robot_self_candidate_ids":[],"unresolved_cube_ids":[],
 "visibility_notes":[],"uncertainty_reasons":[],"evidence":"describe what is visible"}.
Confidence is 0..1. Set booleans from visible evidence, not from the task request. List uncertainty
reasons when unsure, even if this prevents scene export. Do not infer that the robot is ready to move.
"""
        if getattr(builder, "operator_cleared_workspace", False):
            prompt = operator_cleared_prompt()
        prompt = re.sub(
            r"\bA\b",
            getattr(builder, "target_letter", "A"),
            prompt.replace("A rounded transition", "The rounded transition"),
        )
        prompt = prompt.replace(
            "multiple visible As", f"multiple visible {getattr(builder, 'target_letter', 'A')} glyphs"
        )
        prompt += TABLE_EDGE_SAMPLING_GUIDANCE
        return self._call("table_scene", prompt, images)
