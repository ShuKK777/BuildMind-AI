# Demo 0.2 implementation and verification

Version 0.1 remains intact in `demo/` and `archive/demo-v0.1`. Version 0.2 lives in `demo-v0.2/`; root entry opens the new workbench.

## Original Northern Metropolis basis

The supplied `(1).zip` was converted again with the existing authoring converter. Its generated `print-assembly.js` is byte-identical to the assembly already in the repository: SHA-256 `56e5e820df2c3b37d787ef7e243297060a958ad5532d66a9430fd6c3742e1358`. All 27 source parts retain their vertices. All 19 after-development lots retain the manifest positions and dimensions. A02 remains the original default landmark; replacement is a draft until applied and the original can be restored.

The terrain uses the original analytical height field and water layout. The presentation hides the print base/skirt and extends landscape beyond the original boundary. Source millimetres × 1.5 are presentation units, not surveyed metres. No georeferencing or official development metrics are claimed.

## Rendering and workflow

PBR material regrouping, local procedural facade textures, landscape planting, lane markings, street lamps, physical sky/environment reflection, soft shadows and eased cameras enrich the original scene. The workbench adds shared plan/design state, three landmark candidates, editing/undo/apply, project JSON, bilingual UI, local document preview and the 45-second showcase. Exiting the showcase restores the prior project.

Valdrada source is actually imported at runtime, pinned and hashed, with MIT attribution. The full NYC scene, third-party actor assets and Unreal runtime are not included. Building generation is procedural, not image-to-3D inference; review findings are sample project rules, not automatic statutory review.

## Verification

- `npm test`: 15 checks pass, including source geometry fidelity, source layout, real generator outputs, deterministic variants, draft/application state, project round-trip, 45-second clock, local imports, server boundaries and upstream hashes.
- `npm run test:v01`: 13 checks pass; original demo files have no changes.
- Browser: original restoration, variant switching, floor change/application, English/Chinese UI, review source highlighting, JSON export/reimport, complete 45-second showcase and restoration of the pre-showcase design verified.
- Browser screenshots use the CPU compatibility renderer because WebGL is unavailable in the managed browser. GPU materials, environment reflection, shadows and accelerated frame rate require final acceptance on the presentation computer. CPU rendering is an approximate fallback, not a photorealistic or high-frame-rate acceptance result.

Run instructions and capability boundaries are in `demo-v0.2/README.md`.
