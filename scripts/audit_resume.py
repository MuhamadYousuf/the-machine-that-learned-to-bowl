"""Audit a run's generation log and evolution lineage for gaps and duplicates."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def audit(run_name: str) -> dict:
    run_dir = ROOT / "runs" / run_name
    with (run_dir / "log.csv").open("r", newline="", encoding="utf-8") as handle:
        log_rows = list(csv.DictReader(handle))
    with gzip.open(run_dir / "lineage.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        lineage_rows = list(csv.DictReader(handle))
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    log_generations = [int(row["gen"]) for row in log_rows]
    lineage_generations = [int(row["generation"]) for row in lineage_rows]
    log_counts = Counter(log_generations)
    lineage_counts = Counter(lineage_generations)
    child_ids = [row["child_id"] for row in lineage_rows]
    last_generation = max(log_generations, default=0)
    population_size = int(config["population_size"])
    expected_log = list(range(1, last_generation + 1))
    expected_lineage = list(range(0, last_generation + 1))

    boundaries = [459, 460, 461, 599, 600, 601, 669, 670, 671]
    boundary_rows = {
        str(generation): {
            "log_rows": log_counts[generation],
            "lineage_children": lineage_counts[generation],
        }
        for generation in boundaries
    }
    timings_by_generation = {
        int(row["gen"]): float(row["seconds_per_generation"])
        for row in log_rows
        if row.get("seconds_per_generation")
    }
    timings = list(timings_by_generation.values())
    resume_events = [event for event in config.get("events", []) if event.get("type") == "resume"]
    recovery_events = [event for event in config.get("events", []) if event.get("type") == "process_exit_recovery"]
    log_unique = len(log_generations) == len(set(log_generations))
    log_contiguous = sorted(log_generations) == expected_log
    lineage_contiguous = sorted(set(lineage_generations)) == expected_lineage
    lineage_counts_correct = all(lineage_counts[generation] == population_size for generation in expected_lineage)
    child_ids_unique = len(child_ids) == len(set(child_ids))
    around_460_ok = all(
        boundary_rows[str(generation)]["log_rows"] == 1
        and boundary_rows[str(generation)]["lineage_children"] == population_size
        for generation in (459, 460, 461)
    )
    integrity_ok = log_unique and log_contiguous and lineage_contiguous and lineage_counts_correct and child_ids_unique

    result = {
        "resume_state_generation_at_user_restart": 460,
        "run_status": config.get("status"),
        "last_completed_generation": last_generation,
        "verified_through_generation": last_generation,
        "log_rows": len(log_rows),
        "log_unique": log_unique,
        "log_contiguous_from_1": log_contiguous,
        "lineage_rows": len(lineage_rows),
        "lineage_contiguous_from_0": lineage_contiguous,
        "lineage_population_size_per_generation": lineage_counts_correct,
        "lineage_child_ids_unique": child_ids_unique,
        "gen460_resume_event_present": any(int(event.get("generation", -1)) == 460 for event in resume_events),
        "gen459_460_461_boundary_ok": around_460_ok,
        "boundary_rows": boundary_rows,
        "resume_events": resume_events,
        "process_exit_recoveries": recovery_events,
        "timing": {
            "samples": len(timings),
            "sum_seconds_per_generation": sum(timings),
            "mean_seconds_per_generation": sum(timings) / len(timings) if timings else None,
            "sum_seconds_after_user_resume_gen460": sum(
                seconds for generation, seconds in timings_by_generation.items() if generation > 460
            ),
            "sum_seconds_after_recovery_gen670": sum(
                seconds for generation, seconds in timings_by_generation.items() if generation > 670
            ),
        },
        "integrity_ok": integrity_ok,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = run_dir / "resume_audit.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="ga2", help="run folder name under runs/")
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))


if __name__ == "__main__":
    main()
