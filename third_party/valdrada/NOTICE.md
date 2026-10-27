# Valdrada source attribution

This project includes code from **Valdrada**, developed by **Mehmet Kerem Turkcan**:

- Upstream: https://github.com/mkturkcan/valdrada
- Pinned commit: `5dacc4ae0a34ee54fe70d4706cdf3f6662d596d2`
- Copyright (c) 2026 Mehmet Kerem Turkcan
- Code licence: **MIT**, reproduced in [LICENSE](LICENSE).

The files listed in `SOURCE.json` are unmodified upstream source. Their SHA-256 hashes are checked by the project's tests. They include the independent building generator, geometry and batching helpers, component and material libraries, seeded random number generator, procedural textures, and traffic signal phases.

BuildMind adaptations are in `packages/`, outside this directory:

- `building-kit/adapter.js`: explicit landmark families, bounded design input, deterministic massing selection, semantic IDs, actual geometry envelope measurements and lot fitting.
- `showcase/camera-rig.js`: original-print-coordinate adaptation of the camera interpolation/easing design in upstream `core/sim/pathcam.mjs`; the adaptation carries its attribution header.
- `scenario/life.js`: our concept-road routes and lightweight vehicle motion using the original upstream signal-phase function. This is not a port of the full NYC traffic or pedestrian simulator.

The Northern Waterfront district, UI, project state, review integration, massing estimates, curve landmark, lightweight actors and compatibility rendering are BuildMind additions. The district reuses the user-supplied original Northern Metropolis print assembly meshes, positions and terrain formula. Only editable A02 candidates use the building generators; original surrounding buildings retain their source geometry. Presentation materials, facade textures, vegetation, continuous exterior landscape and CPU projections are BuildMind additions.

No compiled NYC city tiles, OSM-derived data modules, third-party character/vehicle model banks, upstream photographs or Unreal runtime are included. Source-code licensing does not grant a blanket licence to those separate assets; refer to upstream `LICENSING.md` if adding them later.
