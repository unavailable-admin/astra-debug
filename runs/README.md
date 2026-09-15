# Runtime output and legacy inputs

New runs save images, videos, simulator state, API responses and reports here.
Generated output is kept locally and ignored by Git.

The allowlist in `.gitignore` preserves existing code dependencies:

- `reset_scene11_worker.py`: cold reset of idle Scene11 on port 8081.
- `run_ace_fresh.py`, `run_ace_trajectory_fresh.py`: historical ACE entry points.
- `initial.jpg`, `scene_authored_geometry.json`: legacy monocular tracker inputs,
  offline test fixtures and authored-geometry accuracy reference.
- `hi_plan_response.json`, `hi_grasp_final.jpg` and
  `hi_demo_20260910_082128/frame_1556.jpg`: historical HI demo/API probe inputs.

The stereo controller obtains current letter positions from stereo images;
its grasp positions do not use the authored object geometry.
Historical recording paths in documentation refer to the original machine.
A fresh clone does not contain those videos or raw API logs.
