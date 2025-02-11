# ruff: noqa
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
from prettytable import PrettyTable

SCENARIO_TO_SEQS = {
    "stationary": ["0099", "0101", "0103", "0106", "0108", "0278", "0331", "0783", "0796", "0966"],
    "frontal": ["0103", "0106", "0110", "0346", "0923"],
    "side": ["0103", "0108", "0110", "0278", "0921"],
}


def _load_metrics_file(folder: Path) -> dict:
    fp = folder / "metrics.json"
    if not fp.exists():
        raise FileNotFoundError(f"Metrics file {fp} not found.")
    with Path.open(fp, "r") as f:
        return json.load(f)


def gather_results(
    results_folder: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    metrics_per_scenario_and_seq = {k: {seq: {} for seq in seqs} for k, seqs in SCENARIO_TO_SEQS.items()}
    metric_keys = set()
    for scenario, seqs in SCENARIO_TO_SEQS.items():
        for seq in seqs:
            runs_folder = results_folder / f"{scenario}-{seq}"
            for run in runs_folder.iterdir():
                if run.is_dir() and (run / "metrics.json").exists():
                    metrics_per_scenario_and_seq[scenario][seq][run.name] = _load_metrics_file(run)
                    for key in metrics_per_scenario_and_seq[scenario][seq][run.name]:
                        if "info" in key:
                            continue
                        if metrics_per_scenario_and_seq[scenario][seq][run.name][key] is None:
                            continue
                        metric_keys.add(key)

    # scenario -> seq -> metric -> value
    avg_metrics_per_scenario_and_seq: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    std_metrics_per_scenario_and_seq: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for scenario, seqs in metrics_per_scenario_and_seq.items():
        for seq in seqs:
            for metric in metric_keys:
                values = [
                    metrics_per_scenario_and_seq[scenario][seq][run][metric]
                    for run in metrics_per_scenario_and_seq[scenario][seq]
                    if metrics_per_scenario_and_seq[scenario][seq][run][metric] is not None
                ]
                avg_metrics_per_scenario_and_seq[scenario][seq][metric] = np.round(np.array(values).mean(), 3)
                std_metrics_per_scenario_and_seq[scenario][seq][metric] = np.round(np.array(values).std(), 3)

    # scenario -> metric -> value
    avg_metrics_per_scenario = defaultdict(dict)
    std_metrics_per_scenario = defaultdict(dict)
    for scenario, seqs in avg_metrics_per_scenario_and_seq.items():
        for metric in metric_keys:
            values = [seq[metric] for seq in seqs.values() if seq[metric] is not None and not np.isnan(seq[metric])]
            avg_metrics_per_scenario[scenario][metric] = np.round(np.array(values).mean(), 3)
            std_metrics_per_scenario[scenario][metric] = np.round(np.array(values).std(), 3)

    # metric -> value
    avg_metrics = {}
    std_metrics = {}
    for metric in metric_keys:
        values = np.array([sc[metric] for sc in avg_metrics_per_scenario.values()])
        values = values[~np.isnan(values)]
        avg_metrics[metric] = np.round(values.mean(), 3)
        std_metrics[metric] = np.round(values.std(), 3)

    return (
        {"avg": avg_metrics, "std": std_metrics},
        {"avg": avg_metrics_per_scenario, "std": std_metrics_per_scenario},
        {"avg": avg_metrics_per_scenario_and_seq, "std": std_metrics_per_scenario_and_seq},
    )


def check_validity(result_path: Path) -> bool:
    stationary_folder_names = [f"stationary-{seq}" for seq in SCENARIO_TO_SEQS["stationary"]]
    frontal_folder_names = [f"frontal-{seq}" for seq in SCENARIO_TO_SEQS["frontal"]]
    side_folder_names = [f"side-{seq}" for seq in SCENARIO_TO_SEQS["side"]]
    folder_names = stationary_folder_names + frontal_folder_names + side_folder_names

    error_message = 0
    # assert all folders are present
    for folder_name in folder_names:
        if not (result_path / folder_name).exists():
            print(f"Folder {folder_name} not found at root: {result_path}.")
            error_message += 1

    # assert all folders have the same number of runs
    n_runs = len(list((result_path / folder_names[0]).iterdir()))
    for folder_name in folder_names:
        if len(list((result_path / folder_name).iterdir())) != n_runs:
            print(f"Folder {folder_name} does not have the same number of runs as the other folders.")
            error_message += 1

        # assert that the metrics file is present in all runs
        for run in (result_path / folder_name).iterdir():
            if not (run / "metrics.json").exists():
                print(f"Metrics file not found in {run}.")
                error_message += 1

    return error_message == 0


def main(result_path: Path, no_check: bool = False) -> None:
    if not check_validity(result_path) and not no_check:
        raise ValueError(f"Invalid results folder: {result_path}.")

    all_table = PrettyTable()
    scenario_table = {scenario: PrettyTable() for scenario in SCENARIO_TO_SEQS}
    scenario_seq_table = {scenario: {seq: PrettyTable() for seq in seqs} for scenario, seqs in SCENARIO_TO_SEQS.items()}

    keys_we_want = [
        "ncap_score",
        "any_collide@0.0s",
        "trajectory_mean_deviation",
        "trajectory_max_deviation",
        "progress_toward_goal",
        "final_goal_distance",
    ]
    header_keys = [k.replace("any_collide", "CR") for k in keys_we_want]

    all_metrics, scenario_metrics, scenario_seq_metrics = gather_results(result_path)

    # first lets print the per scenario and per sequence metrics
    for scenario in scenario_seq_metrics["avg"]:
        for seq in scenario_seq_metrics["avg"][scenario]:
            scenario_seq_table[scenario][seq].field_names = [f"{scenario}-{seq}", *header_keys]
            scenario_seq_table[scenario][seq].add_row(
                [
                    "model",
                    *[
                        f"{scenario_seq_metrics['avg'][scenario][seq].get(k, 0.0)}"
                        "\u00B1"
                        f"{scenario_seq_metrics['std'][scenario][seq].get(k, 0.0)}"
                        for k in keys_we_want
                    ],
                ]
            )

    for scenario in scenario_metrics["avg"]:
        scenario_table[scenario].field_names = [f"{scenario}", *header_keys]
        scenario_table[scenario].add_row(
            [
                "model",
                *[
                    f" {scenario_metrics['avg'][scenario].get(k, 0.0)}"
                    "\u00B1"
                    f"{scenario_metrics['std'][scenario].get(k, 0.0)}"
                    for k in keys_we_want
                ],
            ]
        )

    all_table.field_names = ["Overall", *header_keys]
    all_table.add_row(
        ["model", *[f"{all_metrics['avg'].get(k, 0.0)}" "\u00B1" f"{all_metrics['std'].get(k, 0.0)}" for k in keys_we_want]]
    )

    # print the per sequence tables
    for scenario in scenario_seq_table:
        for seq in scenario_seq_table[scenario]:
            print(scenario_seq_table[scenario][seq])
            print("\n\n")

    print("=" * 100)
    print("=" * 100)
    print("\n\n")

    # print the per scenario tables
    for scenario in scenario_table:
        print(scenario_table[scenario])
        print("\n\n")

    print("=" * 100)
    print("=" * 100)
    print("\n\n")

    # print the overall table
    print(all_table)
    print("\n\n")


if __name__ == "__main__":
    from vam.utils import expand_path

    def boolean_flag(arg: Union[str, bool]) -> bool:
        """Add a boolean flag to argparse parser."""
        if isinstance(arg, bool):
            return arg
        if arg.lower() in ("true", "1", "yes", "y"):
            return True
        elif arg.lower() in ("false", "0", "no", "n"):
            return False
        else:
            raise ValueError(f"Expected 'true'/'false' or '1'/'0', but got '{arg}'")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=expand_path,
        help="Path to the directory containing all result files (typically outputs/<date>)",
    )
    parser.add_argument(
        "--no_check",
        type=boolean_flag,
        default=False,
        help="Disable validity checks (for partial result while still running eval)",
    )
    args = parser.parse_args()
    main(Path(args.rootdir), args.no_check)
