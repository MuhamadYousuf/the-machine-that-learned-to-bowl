import gzip
import json
from pathlib import Path

import numpy as np

from aibowler.dashboard_data import read_csv_tolerant, read_jsonl_tolerant
from aibowler.env import OverEnv
from aibowler.evolve import (
    _make_scenarios,
    load_normalization_baselines,
    mutation_sigma,
    normalize_reward,
    validation_seed_sets,
)
from aibowler.evolve import run_evolution
from aibowler.replay import REPLAY_SCHEMA_VERSION, load_replay
from aibowler.brain_explorer import _checkpoint_path, nearest_checkpoint_name


def test_normalized_fitness_maps_random_to_zero_and_specialist_to_one():
    random_reward = -16.5
    specialist_reward = 4.5
    assert normalize_reward(random_reward, random_reward, specialist_reward) == 0.0
    assert normalize_reward(specialist_reward, random_reward, specialist_reward) == 1.0
    assert normalize_reward((random_reward + specialist_reward) / 2, random_reward, specialist_reward) == 0.5


def test_champion_length_js_divergence_is_symmetric_and_bounded():
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from analyze_run import jensen_shannon_divergence

    first = {"yorker": 1.0, "full": 0.0}
    same = {"yorker": 1.0, "full": 0.0}
    opposite = {"yorker": 0.0, "full": 1.0}
    assert jensen_shannon_divergence(first, same) == 0.0
    assert jensen_shannon_divergence(first, opposite) == 1.0
    assert jensen_shannon_divergence(first, opposite) == jensen_shannon_divergence(opposite, first)


def test_atomic_snapshot_replace_retries_windows_sharing_violation(tmp_path, monkeypatch):
    import aibowler.evolve as evolve

    source = tmp_path / "source.tmp"
    target = tmp_path / "snapshot.npz"
    source.write_bytes(b"complete snapshot")
    real_replace = evolve.os.replace
    calls = {"count": 0}

    def fail_once_with_windows_share_error(src, dst):
        calls["count"] += 1
        if calls["count"] == 1:
            error = PermissionError(13, "file is temporarily open")
            error.winerror = 32
            raise error
        return real_replace(src, dst)

    monkeypatch.setattr(evolve.os, "replace", fail_once_with_windows_share_error)
    monkeypatch.setattr(evolve.time, "sleep", lambda _seconds: None)
    evolve._replace_with_retry(source, target)
    assert target.read_bytes() == b"complete snapshot"
    assert calls["count"] == 2


def test_plan_matrix_random_and_specialist_baselines_normalize_to_endpoints():
    matrix_path = Path(__file__).resolve().parents[1] / "outputs" / "batter" / "plan_matrix.csv"
    baselines = load_normalization_baselines(matrix_path)
    for baseline in baselines.values():
        assert abs(
            normalize_reward(
                baseline["random"], baseline["random"], baseline["specialist"]
            )
        ) < 1e-12
        assert abs(
            normalize_reward(
                baseline["specialist"], baseline["random"], baseline["specialist"]
            )
            - 1.0
        ) < 1e-12


def test_validation_seed_namespace_never_overlaps_training_seed_range():
    training = _make_scenarios(np.random.default_rng(77), 16)
    training_seeds = {seed for _, seed in training}
    validation = validation_seed_sets(77, 200)
    validation_values = {seed for values in validation.values() for seed in values}
    assert validation_values.isdisjoint(training_seeds)
    assert min(validation_values) > np.iinfo(np.uint32).max


def test_dashboard_csv_reader_skips_partial_last_line(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("gen,fitness\n1,0.25\n2,0.", encoding="utf-8")
    assert read_csv_tolerant(path) == [{"gen": "1", "fitness": "0.25"}]


def test_dashboard_jsonl_reader_skips_partial_last_record(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text('{"generation":25,"score":0.4}\n{"generation":50,', encoding="utf-8")
    assert read_jsonl_tolerant(path) == [{"generation": 25, "score": 0.4}]


def test_brain_explorer_resolves_dense_best_genomes_and_nearest_saved_generation(tmp_path):
    best = tmp_path / "best_genomes"
    checkpoints = tmp_path / "checkpoints"
    best.mkdir(); checkpoints.mkdir()
    (best / "gen_0001.npz").touch()
    (best / "gen_0100.npz").touch()
    (checkpoints / "gen_0050.npz").touch()
    assert nearest_checkpoint_name(tmp_path, 97) == ("gen_0100", 100)
    assert nearest_checkpoint_name(tmp_path, 73) == ("gen_0050", 50)
    assert _checkpoint_path(tmp_path, "gen_0100").parent.name == "best_genomes"
    assert _checkpoint_path(tmp_path, "gen_0050").parent.name == "checkpoints"


def test_decision_map_length_categories_use_noise_free_physics_and_cache(tmp_path):
    from aibowler.brain_explorer import decision_map
    from aibowler.policy import MLPPolicy

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    metadata = {"generation": 1, "noise_scale": 0.0, "environment_version": 2}
    np.savez_compressed(
        checkpoints / "gen_0001.npz",
        genome=np.zeros(MLPPolicy.genome_length()),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    result = decision_map(
        tmp_path, "gen_0001", "slogger", 91, 0, 5, 7,
        "length_category", resolution=8,
    )
    repeated = decision_map(
        tmp_path, "gen_0001", "slogger", 91, 0, 5, 7,
        "length_category", resolution=8,
    )

    assert repeated is result
    assert result["unit"] == "length category · noise-free physics"
    assert result["color_center"] == 0.0
    assert set(result["category_labels"]) == {"-3", "-2", "-1", "1", "2", "3"}
    assert len(result["values"]) == 8 and len(result["values"][0]) == 8
    assert {value for row in result["values"] for value in row} <= {-3, -2, -1, 1, 2, 3}


def test_mutation_sigma_uses_cosine_annealing():
    assert mutation_sigma(1, 1500) == 0.08
    assert abs(mutation_sigma(1500, 1500) - 0.015) < 1e-12
    assert mutation_sigma(750, 1500) > 0.015


def test_replay_loader_accepts_gzip_schema_two(tmp_path):
    path = tmp_path / "replay.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump({"schema_version": REPLAY_SCHEMA_VERSION, "overs_data": []}, file)
    assert load_replay(path)["schema_version"] == 2


def test_delivery_info_has_event_times_and_fine_trajectory_samples():
    env = OverEnv(noise_scale=0.0)
    obs = env.reset("slogger", np.random.default_rng(99))
    _, _, _, info = env.step(np.array([0.0, -0.05, 0.0, 0.0]))
    times = [point["t_s"] for point in info["trajectory_points"]]
    assert max(np.diff(times)) <= 0.005
    assert set(info["event_times"]) == {
        "release_s", "bounce_s", "crease_s", "stumps_s",
        "bat_contact_s", "bat_contact_is_crease_proxy",
    }


def test_exact_crash_resume_matches_straight_deterministic_run(tmp_path, monkeypatch):
    import csv
    import json
    import aibowler.evolve as evolve
    from aibowler.env import ARCHETYPE_NAMES

    ticks = {"value": 0}

    def deterministic_clock():
        ticks["value"] += 1
        return float(ticks["value"])

    monkeypatch.setattr(evolve.time, "perf_counter", deterministic_clock)
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["archetype", "plan", "expected_fitness"])
        writer.writeheader()
        for archetype in ARCHETYPE_NAMES:
            writer.writerow({"archetype": archetype, "plan": "P6", "expected_fitness": -10})
            writer.writerow({"archetype": archetype, "plan": "ORACLE", "expected_fitness": 10})
    settings = {
        "generations": 30,
        "population_size": 12,
        "seed": 8181,
        "workers": 1,
        "overs_per_archetype": 1,
        "validation_overs": 2,
        "plan_matrix_path": matrix,
        "save_artifacts": False,
    }
    straight_dir = tmp_path / "straight"
    split_dir = tmp_path / "split"
    straight = run_evolution(straight_dir, **settings)
    interrupted = run_evolution(split_dir, **settings, stop_after_generation=20)
    assert interrupted["last_completed_generation"] == 20
    resumed = run_evolution(split_dir, **settings, exact_resume=True)

    assert np.array_equal(straight["final_population"], resumed["final_population"])
    with (straight_dir / "log.csv").open(newline="", encoding="utf-8") as file:
        straight_rows = list(csv.DictReader(file))
    with (split_dir / "log.csv").open(newline="", encoding="utf-8") as file:
        resumed_rows = list(csv.DictReader(file))
    assert straight_rows == resumed_rows
    with np.load(straight_dir / "checkpoints" / "champion.npz", allow_pickle=False) as first:
        straight_genome = first["genome"].copy()
        straight_metadata = json.loads(str(first["metadata_json"].item()))
    with np.load(split_dir / "checkpoints" / "champion.npz", allow_pickle=False) as second:
        resumed_genome = second["genome"].copy()
        resumed_metadata = json.loads(str(second["metadata_json"].item()))
    assert np.array_equal(straight_genome, resumed_genome)
    assert straight_metadata["validation_score"] == resumed_metadata["validation_score"]
    assert straight_metadata["validation_generation"] == resumed_metadata["validation_generation"]
    assert len({row["gen"] for row in resumed_rows}) == 30

    for generation in range(1, 31):
        with np.load(straight_dir / "best_genomes" / f"gen_{generation:04d}.npz", allow_pickle=False) as first:
            straight_snapshot = first["genome"].copy()
            straight_id = int(first["genome_id"])
        with np.load(split_dir / "best_genomes" / f"gen_{generation:04d}.npz", allow_pickle=False) as second:
            assert np.array_equal(straight_snapshot, second["genome"])
            assert straight_id == int(second["genome_id"])
    with np.load(straight_dir / "populations" / "population_0025.npz", allow_pickle=False) as first:
        straight_population_25 = first["population"].copy()
        straight_fitness_25 = first["fitness"].copy()
        straight_ids_25 = first["genome_ids"].copy()
    with np.load(split_dir / "populations" / "population_0025.npz", allow_pickle=False) as second:
        assert np.array_equal(straight_population_25, second["population"])
        assert np.array_equal(straight_fitness_25, second["fitness"])
        assert np.array_equal(straight_ids_25, second["genome_ids"])


def test_evolution_snapshots_do_not_change_training_results(tmp_path, monkeypatch):
    import csv
    import aibowler.evolve as evolve
    from aibowler.env import ARCHETYPE_NAMES

    ticks = {"value": 0}

    def deterministic_clock():
        ticks["value"] += 1
        return float(ticks["value"])

    monkeypatch.setattr(evolve.time, "perf_counter", deterministic_clock)
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["archetype", "plan", "expected_fitness"])
        writer.writeheader()
        for archetype in ARCHETYPE_NAMES:
            writer.writerow({"archetype": archetype, "plan": "P6", "expected_fitness": -10})
            writer.writerow({"archetype": archetype, "plan": "ORACLE", "expected_fitness": 10})
    settings = {
        "generations": 30,
        "population_size": 8,
        "seed": 9091,
        "workers": 1,
        "overs_per_archetype": 1,
        "validation_overs": 1,
        "plan_matrix_path": matrix,
        "save_artifacts": False,
    }
    with_dir = tmp_path / "with_snapshots"
    without_dir = tmp_path / "without_snapshots"
    with_result = run_evolution(with_dir, **settings, save_evolution_snapshots=True)
    without_result = run_evolution(without_dir, **settings, save_evolution_snapshots=False)

    assert np.array_equal(with_result["final_population"], without_result["final_population"])
    assert with_result["history"] == without_result["history"]
    with (with_dir / "log.csv").open(newline="", encoding="utf-8") as file:
        with_rows = list(csv.DictReader(file))
    with (without_dir / "log.csv").open(newline="", encoding="utf-8") as file:
        without_rows = list(csv.DictReader(file))
    assert with_rows == without_rows
    assert len(list((with_dir / "best_genomes").glob("gen_*.npz"))) == 30
    assert not (without_dir / "best_genomes").exists()
    with np.load(with_dir / "populations" / "population_0025.npz", allow_pickle=False) as snapshot:
        assert snapshot["population"].shape == (8, snapshot["population"].shape[1])
        assert snapshot["fitness"].shape == (8,)
        assert snapshot["genome_ids"].shape == (8,)
    assert not (without_dir / "populations" / "population_0025.npz").exists()
