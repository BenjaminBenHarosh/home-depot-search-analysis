"""Read-only browser for pipeline run outputs under outputs/runs/."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# Files that indicate a completed stage (full-pipeline or a subcommand)
KNOWN_FILES = (
    "results_summary.json",
    "config_used.json",
    "tune_best_params.json",
    "model_comparison.csv",
    "feature_search_results.csv",
    "feature_set_evaluation_results.csv",
    "logs/run.log",
)


def list_runs(output_root: Path) -> list[dict]:
    """Each entry: id, run_dir, summary_path (optional), stage_hint from config."""
    runs_dir = output_root / "runs"
    if not runs_dir.is_dir():
        return []

    rows: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        has_marker = any((run_dir / name).is_file() for name in KNOWN_FILES)
        if not has_marker:
            continue

        summary_path = run_dir / "results_summary.json"
        cmd = None
        cfg_path = run_dir / "config_used.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                cmd = cfg.get("command")
            except (json.JSONDecodeError, OSError):
                cmd = None

        rows.append(
            {
                "id": run_dir.name,
                "run_dir": run_dir,
                "summary_path": summary_path if summary_path.is_file() else None,
                "command": cmd,
            }
        )
    return rows


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    st.set_page_config(page_title="Home Depot runs", layout="wide")
    st.title("Home Depot search analysis — run summaries")
    st.caption(
        "Read-only view of **`outputs/runs/<run_id>/`**. Full pipeline runs include "
        "`results_summary.json`; other CLI stages write `config_used.json` and stage CSVs/JSON."
    )

    default_root = Path("outputs")
    root = Path(st.text_input("Output directory (contains `runs/`)", value=str(default_root))).expanduser()

    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        st.warning(
            f"No **`runs/`** directory at `{runs_dir.resolve()}`. "
            "Use the same `--output-dir` you pass to the CLI (often `outputs`). "
            "Example: `python cli.py run baseline --output-dir outputs` "
            "or `python cli.py run full-pipeline --data-dir ... --output-dir outputs`."
        )
        return

    runs = list_runs(root)
    if not runs:
        st.info(
            f"`runs/` exists at `{runs_dir.resolve()}` but no run folders with pipeline artifacts yet "
            "(looking for files such as `results_summary.json`, `config_used.json`, …). "
            "**Tip:** run any stage, e.g. `python cli.py run tune --output-dir outputs`, "
            "then refresh this page."
        )
        return

    labels = []
    for r in runs:
        tag = "full pipeline" if r["summary_path"] else "stage run"
        cmd = r["command"]
        extra = f" — {cmd}" if cmd else ""
        labels.append(f"{r['id']} ({tag}){extra}")

    pick_label = st.selectbox("Run", options=labels, index=0)
    idx = labels.index(pick_label)
    picked = runs[idx]

    run_dir = picked["run_dir"]
    summary_path = picked["summary_path"]

    cfg_path = run_dir / "config_used.json"

    if summary_path is not None:
        data = load_summary(summary_path)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary")
            st.json({k: v for k, v in data.items() if k != "artifacts"})
        with col2:
            st.subheader("Artifact paths")
            if "artifacts" in data:
                st.json(data["artifacts"])
            st.code(str(summary_path.resolve()), language="text")
    else:
        cmd = picked.get("command") or "a pipeline stage"
        st.subheader("Run configuration")
        st.info(
            f"This run was produced by **`{cmd}`** only. That is normal: "
            "`results_summary.json` (schema-validated metrics bundle) is written **only** by "
            "`python cli.py run full-pipeline`. Use the files and log below for this stage."
        )
        if cfg_path.is_file():
            st.json(json.loads(cfg_path.read_text(encoding="utf-8")))
        else:
            st.caption("No `config_used.json` in this folder.")

    log_path = run_dir / "logs" / "run.log"
    if log_path.is_file():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-120:])
            with st.expander("run.log (last 120 lines)", expanded=False):
                st.code(tail or "(empty)", language="text")
        except OSError:
            pass

    if summary_path is not None and cfg_path.is_file():
        with st.expander("config_used.json"):
            st.json(json.loads(cfg_path.read_text(encoding="utf-8")))

    with st.expander("Files in this run directory"):
        lines = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
        st.code("\n".join(lines) if lines else "(empty)", language="text")


if __name__ == "__main__":
    main()
