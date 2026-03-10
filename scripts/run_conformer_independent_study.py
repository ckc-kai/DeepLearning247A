import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


RE_VAL_CER = re.compile(r"(?:'val/CER':\s*|val/CER\s*│\s*)([0-9.]+)")
RE_MEAN_P_BLANK = re.compile(r"mean_P\(blank\)=([0-9.]+)")


COMMON_TRIAL = {
    "batch_size": 8,
    "lr": 1e-4,
    "warmup_epochs": 5,
    "mlp_features": "[128]",
    "num_layers": 6,
    "n_heads": 4,
    "ffn_expansion": 2,
    "conv_kernel_size": 15,
    "dropout": 0.1,
    "blank_logit_sub": 3.0,
    "blank_sub_anneal_epochs": 8,
    "time_reduction_stride": 2,
    "time_reduction_kernel_size": 3,
    "local_attention_window": 32,
    "entropy_weight": 0.0,
    "intermediate_ctc_weight": 0.0,
    "intermediate_ctc_layer": 0,
}


CANDIDATE_GROUPS = {
    "stability": [
        {
            "name": "stable_minimum",
            "description": "Minimal stable Conformer: no pretraining, no LM, no refinement, no iCTC.",
        },
    ],
    "depth": [
        {
            "name": "depth4",
            "description": "Depth ablation: shallower 4-layer Conformer.",
            "num_layers": 4,
        },
        {
            "name": "depth6",
            "description": "Depth ablation anchor: 6-layer stable minimum.",
        },
        {
            "name": "depth8",
            "description": "Depth ablation: deeper 8-layer Conformer.",
            "num_layers": 8,
            "batch_size": 4,
        },
    ],
    "width": [
        {
            "name": "width96",
            "description": "Width ablation: narrower frontend width.",
            "mlp_features": "[96]",
        },
        {
            "name": "width128",
            "description": "Width ablation anchor: 128-d frontend width.",
        },
        {
            "name": "width160",
            "description": "Width ablation: wider frontend width within 6GB budget.",
            "mlp_features": "[160]",
            "batch_size": 4,
        },
    ],
    "kernel": [
        {
            "name": "kernel15",
            "description": "Kernel ablation anchor: conv kernel 15.",
        },
        {
            "name": "kernel31",
            "description": "Kernel ablation: larger local conv kernel.",
            "conv_kernel_size": 31,
        },
    ],
    "locality": [
        {
            "name": "global_attn",
            "description": "Locality ablation: global attention without local window.",
            "local_attention_window": 0,
        },
        {
            "name": "local32",
            "description": "Locality ablation anchor: local attention window 32.",
        },
    ],
}


PHASES = {
    "sanity": {
        "max_epochs": 2,
        "limit_train_batches": 80,
        "limit_val_batches": 4,
        "limit_test_batches": 1,
    },
    "screening": {
        "max_epochs": 5,
        "limit_test_batches": 1,
    },
    "final": {
        "max_epochs": 40,
    },
}


def parse_metrics(text: str) -> tuple[float | None, float | None]:
    val_cers = RE_VAL_CER.findall(text)
    blank_probs = RE_MEAN_P_BLANK.findall(text)
    val_cer = float(val_cers[-1]) if val_cers else None
    mean_p_blank = float(blank_probs[-1]) if blank_probs else None
    return val_cer, mean_p_blank


def resolve_python(python_override: str | None) -> str:
    if python_override:
        return python_override
    preferred = Path.home() / "miniconda3" / "envs" / "emg2qwerty" / "python.exe"
    if preferred.exists():
        return str(preferred)
    return sys.executable


def build_command(
    run_dir: Path,
    phase: str,
    trial: dict,
    python_executable: str,
) -> list[str]:
    phase_cfg = PHASES[phase]
    overrides = [
        "user=single_user",
        "model=conformer_ctc",
        f"trainer.max_epochs={phase_cfg['max_epochs']}",
        f"batch_size={trial['batch_size']}",
        f"optimizer.lr={trial['lr']}",
        f"lr_scheduler.scheduler.warmup_epochs={trial['warmup_epochs']}",
        f"module.mlp_features={trial['mlp_features']}",
        f"module.num_layers={trial['num_layers']}",
        f"module.n_heads={trial['n_heads']}",
        f"module.ffn_expansion={trial['ffn_expansion']}",
        f"module.conv_kernel_size={trial['conv_kernel_size']}",
        f"module.dropout={trial['dropout']}",
        f"module.blank_logit_sub={trial['blank_logit_sub']}",
        f"module.blank_sub_anneal_epochs={trial['blank_sub_anneal_epochs']}",
        f"module.time_reduction_stride={trial['time_reduction_stride']}",
        f"module.time_reduction_kernel_size={trial['time_reduction_kernel_size']}",
        f"module.local_attention_window={trial['local_attention_window']}",
        f"module.entropy_weight={trial['entropy_weight']}",
        f"module.intermediate_ctc_weight={trial['intermediate_ctc_weight']}",
        f"module.intermediate_ctc_layer={trial['intermediate_ctc_layer']}",
        "module.pretrained_frontend=null",
        "module.freeze_frontend=false",
        f"hydra.run.dir={run_dir.as_posix()}",
        "+trainer.precision=16",
        "+trainer.gradient_clip_val=1.0",
        "+trainer.gradient_clip_algorithm=norm",
        "+trainer.log_every_n_steps=20",
    ]
    if "limit_train_batches" in phase_cfg:
        overrides.append(f"+trainer.limit_train_batches={phase_cfg['limit_train_batches']}")
    if "limit_val_batches" in phase_cfg:
        overrides.append(f"+trainer.limit_val_batches={phase_cfg['limit_val_batches']}")
    if "limit_test_batches" in phase_cfg:
        overrides.append(f"+trainer.limit_test_batches={phase_cfg['limit_test_batches']}")
    return [python_executable, "-m", "emg2qwerty.train", *overrides]


def run_trial(
    repo_root: Path,
    output_root: Path,
    phase: str,
    trial: dict,
    python_executable: str,
    dry_run: bool,
) -> dict:
    run_dir = output_root / trial["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(run_dir, phase, trial, python_executable)
    result = {
        "phase": phase,
        "name": trial["name"],
        "description": trial["description"],
        "command": " ".join(command),
        "returncode": None,
        "success": False,
        "val_cer": None,
        "mean_p_blank": None,
        "oom": False,
    }
    if dry_run:
        return result

    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    combined = completed.stdout + "\n" + completed.stderr
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    val_cer, mean_p_blank = parse_metrics(combined)
    result["returncode"] = completed.returncode
    result["success"] = completed.returncode == 0 and val_cer is not None
    result["val_cer"] = val_cer
    result["mean_p_blank"] = mean_p_blank
    result["oom"] = "out of memory" in combined.lower()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled Conformer study focused on correctness and stability."
    )
    parser.add_argument(
        "--phase",
        choices=sorted(PHASES.keys()),
        required=True,
        help="sanity: 2 epochs, screening: 5 epochs, final: 40 epochs",
    )
    parser.add_argument(
        "--group",
        choices=sorted(CANDIDATE_GROUPS.keys()),
        required=True,
        help="Which controlled ablation group to run.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional candidate names to run. Default runs all candidates in the group.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without launching training.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Optional explicit Python executable for training runs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "logs" / "conformer_independent_study" / args.phase / args.group
    output_root.mkdir(parents=True, exist_ok=True)
    python_executable = resolve_python(args.python)

    selected = [{**COMMON_TRIAL, **trial} for trial in CANDIDATE_GROUPS[args.group]]
    if args.only:
        wanted = set(args.only)
        selected = [trial for trial in selected if trial["name"] in wanted]
        if not selected:
            raise SystemExit(f"No matching candidates for --only: {sorted(wanted)}")

    results = []
    for trial in selected:
        print(f"\n=== {trial['name']} ({args.phase}/{args.group}) ===")
        result = run_trial(
            repo_root,
            output_root,
            args.phase,
            trial,
            python_executable,
            args.dry_run,
        )
        print(result["command"])
        if not args.dry_run:
            print(
                f"success={result['success']} val_cer={result['val_cer']} "
                f"mean_p_blank={result['mean_p_blank']} oom={result['oom']}"
            )
        results.append(result)

    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ranked = [item for item in results if item["val_cer"] is not None]
    ranked.sort(key=lambda item: item["val_cer"])
    print("\n=== Leaderboard ===")
    for idx, item in enumerate(ranked, start=1):
        print(
            f"{idx}. {item['name']}  val/CER={item['val_cer']}  "
            f"mean_P(blank)={item['mean_p_blank']}"
        )
    print(f"\nSaved summary to: {results_path}")


if __name__ == "__main__":
    main()
