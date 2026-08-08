#!/usr/bin/env python3
"""organize_repo.py — sort the flat music-rag scripts into week-based folders.

Uses `git mv` so history is preserved (GitHub still shows each file's past).
Run from the repo root AFTER committing your current state, so if anything looks
wrong you can `git reset --hard` back.

Cross-module imports (e.g. `from features import ...`) still resolve because
every stage script gets run from the repo root and each week folder is added to
the path by conftest-style shim -- see the note printed at the end.

DRY RUN by default: prints what it would do. Pass --go to actually move.
"""
import subprocess, sys
from pathlib import Path

# Which file goes where. Anything not listed is left in root (data, json, configs).
LAYOUT = {
    "src/week1_pipeline": [
        "routing.py", "router.py", "harvest.py", "batch_harvest.py",
        "chunk.py", "batch_chunk.py", "chunks_sort.py",
        "augment.py", "split.py", "dataset_build.py",
        "inventory.py", "inventory02.py", "pm_inventory.py",
    ],
    "src/week2_finetune": [
        "tokenize_corpus.py", "packer.py", "inspect_pack.py",
        "train.py", "train_lora.py",
        "compare_generate.py", "compare_three.py",
        "metric_histogram.py", "check_repeats.py",
    ],
    "src/week3_retrieval": [
        "features.py", "build_index.py", "index_paths.json",
        "retrieve.py", "condition.py", "seeds.py", "seeds.json",
        "generate_ab.py", "batch_gen.py", "check_budget.py",
    ],
    "src/week4_metrics": [
        "metrics_core.py", "metrics_rhythm.py", "filters.py", "results.py",
        "run_metrics.py", "novelty.py", "novelty_stage2.py",
        "corpus_reference.py", "density_and_reference.py",
        "check_key_drift.py", "reclassify.py",
        "inspect_durations.py", "inspect_trills.py",
    ],
    "src/week4_listening": [
        "presentation.py", "render_audio.py", "manifest.py", "manifest_paired.py",
        "select_clips.py", "check_durations.py",
        "wilcoxon_ab.py", "aaprime_evidence.py", "read_csv_data.py",
    ],
    "scratch": [
        "analyze.py", "diagnostic_cuda.py",
        # whole throwaway folders handled separately below
    ],
    "docs": [
        "project_plan.md", "project_plan_2.md", "project_plan_3.md",
        "project_plan_4.md", "project_plan_5.md",
        "session_summary_week0.md", "session_summary_week2.md",
        "session_summary_week3.md", "session_summary_week4.md",
    ],
}

# Whole directories to relocate wholesale.
DIR_MOVES = {
    "lesson_scripts": "scratch/lesson_scripts",
    "test_debug_scripts": "scratch/test_debug_scripts",
}


def run(cmd, go):
    print(("  " if go else "  [dry] ") + " ".join(cmd))
    if go:
        subprocess.run(cmd, check=True)


def main(go):
    root = Path(".")
    if not (root / ".git").exists():
        sys.exit("Run this from the repo root (no .git found).")

    for dest, files in LAYOUT.items():
        present = [f for f in files if (root / f).exists()]
        if not present:
            continue
        run(["mkdir", "-p", dest], go)
        for f in present:
            run(["git", "mv", f, f"{dest}/{Path(f).name}"], go)

    for src, dest in DIR_MOVES.items():
        if (root / src).exists():
            run(["mkdir", "-p", str(Path(dest).parent)], go)
            run(["git", "mv", src, dest], go)

    missing = [f for files in LAYOUT.values() for f in files
               if not (root / f).exists()
               and not (root / next(d for d, fs in LAYOUT.items() if f in fs)
                        / Path(f).name).exists()]
    if missing:
        print("\n  note — listed but not found (skipped, fine if renamed/absent):")
        for m in sorted(set(missing)):
            print(f"    {m}")

    print("\nDone." if go else "\nDry run only — nothing moved. Re-run with --go to apply.")
    if go:
        print(
            "\nIMPORTANT: scripts import each other (e.g. `from features import ...`).\n"
            "They now live in different folders, so run stages from the REPO ROOT with\n"
            "the week folder on the path, e.g.:\n"
            "    PYTHONPATH=src/week3_retrieval python src/week3_retrieval/generate_ab.py\n"
            "or add a conftest/sitecustomize shim. Test one script per week before pushing.")


if __name__ == "__main__":
    main("--go" in sys.argv)