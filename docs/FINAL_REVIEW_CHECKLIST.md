# Final-review completion checklist

## Working application
- [ ] Add `backend/` and `frontend/` to the main repository.
- [ ] Frontend starts with `npm run dev` and displays API health.
- [ ] Backend loads SDXL + IP-Adapter once on the Colab T4.
- [ ] Uploading a reference and generating a single scene works for Fixed, Adaptive-only, and Proposed.
- [ ] Result image, CLIP-I, DINO, CLIP-T, attempt trace, selected scale, retry count, and time render correctly.
- [ ] A 2-4 scene storyboard request completes without a second model load.
- [ ] Optional LLM planner is tested only if API key is configured; otherwise manual prompt mode is demonstrated.
- [ ] Self-correction retry metadata is fixed in the core `storyboard_generator.py`.

## Research reproducibility
- [ ] Preserve `evaluation_results.csv`, calibration data, selected PNGs, and storyboard PDF outside the ephemeral Colab runtime.
- [ ] Confirm metrics and retry fields are defined consistently between the CLI experiment and API.
- [ ] Report DINO/CLIP-I/CLIP-T as proxies; do not claim they guarantee character identity.
- [ ] Clearly report the CLIP-T decrease and generation-time increase as limitations/trade-offs.
- [ ] If formal human evaluation is not conducted, list it as future work rather than claiming it is complete.

## Report / presentation
- [ ] Report Chapters 1-3, implementation/results chapter, conclusion/future work, and references are complete.
- [ ] PPT matches implemented scope; distinguish completed visual generation and correction from optional LLM planning.
- [ ] Demo is rehearsed with a preloaded backend and local image/storyboard backups.
- [ ] Final Git status is clean; no secrets, generated model weights, temporary tokens, or large artifacts committed unintentionally.
