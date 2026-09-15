"""Current-frame candidate identification; the API never supplies world coordinates."""

import asyncio
import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .api import OfficialOpenAI


def mask(image):
    b, g, r = cv2.split(image)
    return ((b > 130) & (g > 100) & (r < 100)).astype(np.uint8) * 255


def parse_json(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class AstraVision:
    def __init__(self, directory, timeout=120):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.client = OfficialOpenAI()

    def _call(self, stage, prompt, images):
        stem = self.directory / f"{time.time_ns()}_{stage}"
        content = [{"type": "text", "text": prompt}]
        for path in images:
            im = Image.open(path).convert("RGB")
            im.thumbnail((960, 540))
            data = io.BytesIO()
            im.save(data, format="JPEG", quality=80)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(data.getvalue()).decode()
                    },
                }
            )
        body = {
            "model": self.client.model,
            "reasoning_effort": "low",
            "max_completion_tokens": 2048,
            "messages": [
                {
                    "role": "system",
                    "content": "Inspect robot simulation images carefully. Return only the requested JSON object. Report uncertainty explicitly. A commanded robot motion does not prove an object moved. Never infer success from the task instruction.",
                },
                {"role": "user", "content": content},
            ],
        }
        stem.with_suffix(".request.json").write_text(
            json.dumps(
                {
                    **self.client.metadata(),
                    "stage": stage,
                    "prompt": prompt,
                    "images": list(map(str, images)),
                },
                indent=2,
            )
        )
        start = time.monotonic()
        try:
            raw = self.client.chat(body, timeout=self.timeout)
            stem.with_suffix(".response.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2)
            )
            if raw["choices"][0].get("finish_reason") != "stop":
                raise ValueError("Incomplete model response")
            result = parse_json(raw["choices"][0]["message"]["content"])
            stem.with_suffix(".decision.json").write_text(
                json.dumps(
                    {"elapsed_seconds": time.monotonic() - start, "decision": result},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return result
        except Exception as exc:
            stem.with_suffix(".error.json").write_text(
                json.dumps(
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "elapsed_seconds": time.monotonic() - start,
                    },
                    indent=2,
                )
            )
            raise

    async def scene_candidates(self, image, word="ACE"):
        """Ask for measured component IDs instead of ambiguous normalized axes."""
        original = Image.open(image).convert("RGB")
        binary = mask(cv2.imread(str(image)))
        (count, _, stats, _) = cv2.connectedComponentsWithStats(binary)
        components = [
            tuple(map(int, stats[i]))
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 20 and max(stats[i, 2:4]) <= 80
        ]
        components.sort(key=lambda item: (item[1] // 25, item[0]))
        candidates = {}
        marked = original.copy()
        draw = ImageDraw.Draw(marked)
        sheet = Image.new("RGB", (6 * 100, max(1, (len(components) + 5) // 6) * 100), "white")
        sd = ImageDraw.Draw(sheet)
        for index, (x, y, w, h, area) in enumerate(components, 1):
            candidates[str(index)] = {"pixel": [x + w / 2, y + h / 2], "bbox": [x, y, w, h]}
            draw.rectangle((x - 2, y - 2, x + w + 2, y + h + 2), outline="red", width=1)
            draw.text((x, y - 13), str(index), fill="red")
            crop = original.crop(
                (
                    max(0, x - 7),
                    max(0, y - 7),
                    min(original.width, x + w + 7),
                    min(original.height, y + h + 7),
                )
            )
            crop.thumbnail((74, 74))
            gx = (index - 1) % 6 * 100
            gy = (index - 1) // 6 * 100
            sheet.paste(crop, (gx + 12, gy + 20))
            sd.text((gx + 8, gy + 4), str(index), fill="black")
        stem = self.directory / f"{time.time_ns()}_candidates"
        marked_path = stem.with_suffix(".scene.jpg")
        sheet_path = stem.with_suffix(".glyphs.jpg")
        marked.save(marked_path)
        sheet.save(sheet_path)
        stem.with_suffix(".map.json").write_text(json.dumps(candidates, indent=2))
        result = await asyncio.to_thread(
            self._call,
            "scene_candidates",
            f'Task: arrange the letter cubes to spell {word} left to right on the empty front row of the table. Image 1 is the current scene with numbered cyan glyph components. Image 2 enlarges the SAME numbered components for identity recognition; its layout is NOT the physical table. Identify visible letters from {list(word)} by candidate_id. Use the full scene to assess location and upright pose. Return JSON {{"letters":[{{"letter":"A","candidate_id":1,"upright":true,"confidence":0.99}}],"row_complete":false,"next_action":{{"action":"pick_place|finish|stop","letter":"A"}},"evidence":"what is visible"}}. Do not output u/v or invent world coordinates. upright means its glyph faces up. Omit missing/occluded letters. Choose pick_place for one visible upright target still in the original letter grid, finish only when all target letters are placed and separate from the hands in the front row in correct order, otherwise stop if uncertain. A robot skill will use the measured candidate center and fixed row targets.',
            [marked_path, sheet_path],
        )
        result["coordinate_mode"] = "candidate_id"
        result["candidate_map"] = candidates
        return result


def identified_pixels(decision, shape, word="ACE"):
    h, w = shape[:2]
    result = {}
    for item in decision.get("letters", []):
        if not isinstance(item, dict):
            continue
        letter = item.get("letter")
        if not isinstance(letter, str) or letter not in word or item.get("upright") is not True:
            continue
        if decision.get("coordinate_mode") == "candidate_id":
            candidate = item.get("candidate_id")
            confidence = item.get("confidence")
            if (
                type(candidate) is not int
                or type(confidence) not in (int, float)
                or not 0.9 <= confidence <= 1
            ):
                continue
            measured = decision.get("candidate_map", {}).get(str(candidate))
            if measured:
                result[letter] = measured["pixel"]
            continue
    return result
