# Experiments

Place new experiment scripts here. Convention:

- Name scripts descriptively: `jlf_loocv.py`, `boundary_refinement.py`, etc.
- Import shared code from `utils/`: `from utils import LABELS, compute_dices, register_pair`
- Use `utils.REPO_ROOT` for repo-relative paths
- Save results to `results/<experiment_name>/`
- Save figures to `figures/`
