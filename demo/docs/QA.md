# Validation record — 2026-10-26

## Automated

`node --test demo/tests/*.test.mjs`: **13 tests passed**.

Covers exact 45-second clock behavior at 12/30/60/120 simulated FPS, pause/resume/restart semantics, monotonic camera easing; persisted-state validation; height changes; 45-second script ordering and module coverage; sample finding/source/evidence mapping; file type/size boundaries; local asset presence; 27-part source identity/coordinates/top-height; A02 being the tallest source tower; immutable original height; SHA-256 checks for 205 bundled dependency files; loopback HTTP GET/HEAD, MIME, write rejection and traversal rejection.

## Browser interaction

Verified in the managed Chromium preview at 1363×936:

- Before stage displays eight village houses and farmland; after stage displays original 19-building assembly.
- Original A02 height 72 mm; all three replacement cards update the scene and details; restore returns the original tower and 72 mm.
- Glass twins standard 86 mm, sky gardens standard 66 mm; height controls update geometry and label.
- Pointer drag rotates the real scene. Clicking a residential building selects A04 and shows its 49.3 mm assembled height.
- Applying a chosen variant returns it to the district; surrounding source meshes remain in place.
- Chinese/English navigation and document sample text switch.
- Showcase paused/resumed, then ran to completion and returned to Plan with Sky Gardens/night state preserved.
- Sample finding 03 expands its evidence and highlights/scrolls to drainage source text.
- DOCX fixture renders text paragraphs, including literal `<script>` text without executing it, with no assigned findings.
- Local PDF fixture renders; Next displays actual second-page content and `2 / 2`, with Next disabled.
- Updated showcase completed all nine cues; its pause state survived a language switch without consuming time.
- Keyboard focus remained on Restart across progress updates and a development-stage change; Enter restarted at the beginning.
- Scene changes use short canvas crossfades, camera moves interpolate with stationary endpoints, daylight blends continuously, and the progress bar updates without recreating playback controls.
- Own-source syntax checks and `git diff --check` passed.

## Limits / target-device check

The managed browser disables WebGL. Browser observations above use the explicit CPU 3D compatibility mode, with shared mesh geometry, camera and ray picking. The compatibility renderer now schedules up to 30 FPS with a smaller raster buffer and cached face normals. This is a scheduling ceiling, not a measured target-device FPS guarantee. Accelerated WebGL lighting/shadows have **not** been visually tested here. Windows/macOS launcher execution, very large or encrypted PDFs, and mobile screen sizes have not been exercised on physical target devices.

Offline readiness is established by shipped local dependencies and source/asset checks. No network-toggle test was available in the managed browser. Before a competition, launch the included server on the actual presentation machine, disconnect its external network, then try the three modules and 45-second playback once.

## Screenshots

![A02 replacement studio](design-preview.jpg)

![Document review and source evidence](review-preview.jpg)
