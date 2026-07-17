# CLAUDE.md

Fork of [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) adding native Apple MLX inference (Apple Silicon) for the full pipeline: Stage 1 shape generation (image → mesh) and Stage 2 PBR texture synthesis (mesh + image → textured GLB).

## Iso-upstream rule (hard constraint)

This fork must stay trivially mergeable with upstream:

- **Never modify upstream PyTorch files.** All MLX code lives in additive `*_mlx.py` files (or `*_mlx/` packages) next to the upstream file they mirror, with the same module/class/method structure so a side-by-side diff shows only PyTorch↔MLX op substitutions.
- Config defaults must match the PyTorch reference exactly. Any deviation hides a port bug — match first, optimize only for a documented framework constraint (e.g. Metal command-buffer budget, handled via tiling).
- **Iso-upstream includes the runtime execution config, not just code structure**: dtype policy (upstream runs both stages in fp16 end-to-end), device semantics, scheduler precision islands. MLX type promotion is stricter than PyTorch's (0-d arrays are not weak like torch scalar tensors), so a single fp32 constant silently upcasts everything downstream — verify actual kernel dtypes with `smeltr record` + op summary after any pipeline change, don't trust the code diff.
- Keep `git fetch upstream` clean: the only tolerated upstream edits are the README MLX section and appended MLX deps in `hy3dshape/requirements.txt`.

## Workflow

- `main` is strictly protected: **no direct pushes**, even for admins. Branch → PR → green `unit-tests` check → merge.
- Commit messages follow **Conventional Commits** (enforced by the `commit-lint` CI job on PRs).
- All repo content — code, comments, docs, config strings — is written in **English**.
- Governance: intendant (`.intendant.toml`, advisory). Docs: `docs/forward_pass.md` (porting principle), `docs/adr/`.

## Tests

```bash
# Unit suite (~2 s, no weights needed) — this is what CI runs
.venv/bin/python -m pytest -q hy3dpaint/tests \
  --ignore=hy3dpaint/tests/test_e2e_paint.py \
  --ignore=hy3dpaint/tests/test_e2e_mesh.py
```

- `.venv` is a uv-managed Python 3.12 env (`uv pip install --python .venv/bin/python <pkg>`); it has no pip.
- e2e scripts (`tests/test_stage1_to_stage2.py`, `hy3dpaint/tests/test_e2e_*.py`) need converted weights (HF `dgrauet/hunyuan3d-2.1-mlx`, override with `HUNYUAN3D_MLX_WEIGHTS_DIR`) and a real GPU — run manually, not in CI.
- Parity vs PyTorch: `hy3dpaint/tests/compare_mlx_pytorch.py` (torch is a dev-only dep). Validated status lives in `hy3dpaint/tests/DEBUGGING_STATUS.md`.
- Root `conftest.py` falls back to the MLX CPU backend when Metal is unavailable (CI runners).
