"""Cached model-inspection data for the local Brain Explorer dashboard."""

from __future__ import annotations

import csv
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .env import ARCHETYPE_NAMES, LEGACY_OBSERVATION_SIZE, OBSERVATION_SIZE, OverEnv
from .evolve import _evaluate_genome_payload, load_normalization_baselines
from .physics import PitchConditions, release_from_action, simulate_delivery
from .policy import ACTION_SIZE, HIDDEN_SIZES, MLPPolicy


INPUT_NAMES = (
    "is Slogger", "is Technician", "is Tailender", "batter confidence",
    "deliveries faced", "legal balls this over", "runs this over",
    "older length", "older line", "older movement", "older speed",
    "older valid", "last length", "last line", "last movement",
    "last speed", "last valid", "bouncers used",
)
OUTPUT_NAMES = ("speed", "elevation", "azimuth", "swing")
INPUT_RANGES = tuple(
    (0.0, 1.0) if index in (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 16, 17)
    else (-1.0, 1.0)
    for index in range(OBSERVATION_SIZE)
)
LAYER_NAMES = ("input → hidden 1", "hidden 1 → hidden 2", "hidden 2 → output")
POPULATION_RUNS = ("ga1500", "ga2", "ga200")


def _checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    name = Path(checkpoint).name
    if name.endswith(".npz"):
        name = name[:-4]
    checkpoint_path = run_dir / "checkpoints" / f"{name}.npz"
    best_path = run_dir / "best_genomes" / f"{name}.npz"
    return checkpoint_path if checkpoint_path.exists() or not best_path.exists() else best_path


def nearest_checkpoint_name(run_dir: Path, generation: int) -> tuple[str, int]:
    """Resolve a requested generation to its nearest durable genome snapshot."""
    choices = []
    for folder in ("best_genomes", "checkpoints"):
        for path in (run_dir / folder).glob("gen_*.npz"):
            try:
                choices.append((int(path.stem.split("_", 1)[1]), path.stem))
            except ValueError:
                continue
    if not choices:
        raise FileNotFoundError("no generation genome snapshots are available")
    saved, name = min(choices, key=lambda item: (abs(item[0] - int(generation)), item[0]))
    return name, saved


@lru_cache(maxsize=96)
def _load_checkpoint_cached(path_text: str, mtime_ns: int, size: int):
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as data:
        genome = data["genome"].astype(float)
        metadata = json.loads(str(data["metadata_json"].item()))
    policy = MLPPolicy(genome)
    w1, b1, w2, b2, w3, b3 = policy._parameters()
    return {
        "genome": genome,
        "metadata": metadata,
        "input_size": policy.input_size,
        "weights": [w1.copy(), w2.copy(), w3.copy()],
        "biases": [b1.copy(), b2.copy(), b3.copy()],
    }


def checkpoint_bundle(run_dir: Path, checkpoint: str) -> dict:
    path = _checkpoint_path(run_dir, checkpoint)
    if not path.exists():
        raise FileNotFoundError(path.name)
    stat = path.stat()
    return _load_checkpoint_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def network_payload(run_dir: Path, checkpoint: str) -> dict:
    bundle = checkpoint_bundle(run_dir, checkpoint)
    return {
        "checkpoint": checkpoint,
        "metadata": bundle["metadata"],
        "input_size": bundle["input_size"],
        "inputs": list(INPUT_NAMES[: bundle["input_size"]]),
        "outputs": list(OUTPUT_NAMES),
        "hidden_sizes": list(HIDDEN_SIZES),
        "weights": [matrix.tolist() for matrix in bundle["weights"]],
        "biases": [vector.tolist() for vector in bundle["biases"]],
    }


def _adapt_observations(policy: MLPPolicy, states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] == LEGACY_OBSERVATION_SIZE and policy.input_size == OBSERVATION_SIZE:
        values = np.pad(values, ((0, 0), (0, 1)))
    elif values.shape[1] == OBSERVATION_SIZE and policy.input_size == LEGACY_OBSERVATION_SIZE:
        values = values[:, :LEGACY_OBSERVATION_SIZE]
    if values.shape[1] != policy.input_size:
        raise ValueError(f"expected {policy.input_size} policy inputs; got {values.shape[1]}")
    return values


def _forward_batch(policy: MLPPolicy, observations: np.ndarray):
    values = _adapt_observations(policy, observations)
    w1, b1, w2, b2, w3, b3 = policy._parameters()
    hidden1 = np.tanh(values @ w1 + b1)
    hidden2 = np.tanh(hidden1 @ w2 + b2)
    raw = hidden2 @ w3 + b3
    return hidden1, hidden2, raw, np.tanh(raw)


@lru_cache(maxsize=96)
def _cam_cached(
    checkpoint_path: str,
    mtime_ns: int,
    size: int,
    archetype: str,
    seed: int,
    noise_scale: float,
    environment_version: int,
) -> dict:
    bundle = _load_checkpoint_cached(checkpoint_path, mtime_ns, size)
    policy = MLPPolicy(bundle["genome"])
    env = OverEnv(noise_scale=noise_scale, environment_version=environment_version)
    observation = env.reset(archetype, np.random.default_rng(seed))
    records = []
    for ball_index in range(24):
        hidden1, hidden2, raw = policy.trace(observation)
        action = np.tanh(raw)
        observation_next, reward, done, info = env.step(action)
        feature = info["features"]
        outcome = info["outcome"]
        records.append({
            "ball": ball_index + 1,
            "observation": observation.astype(float).tolist(),
            "hidden1": hidden1.astype(float).tolist(),
            "hidden2": hidden2.astype(float).tolist(),
            "raw_outputs": raw.astype(float).tolist(),
            "action": action.astype(float).tolist(),
            "release": {
                "speed_kmh": float(info["release"].speed_kmh),
                "elevation_deg": float(info["release"].elevation_deg),
                "azimuth_deg": float(info["release"].azimuth_deg),
                "swing": float(info["release"].swing),
            },
            "delivery": {
                "length_m": 0.0 if feature.length_m is None else float(feature.length_m),
                "line_m": float(feature.line_m),
                "category": feature.length_category,
                "height_at_crease_m": float(feature.height_at_crease_m),
                "result": outcome.result,
                "runs": int(outcome.runs_conceded),
                "wicket": bool(outcome.is_wicket),
                "legal": bool(outcome.is_legal_ball),
                "bouncer_no_ball": bool(info.get("bouncer_no_ball", False)),
            },
            "reward": float(reward),
            "event_times": info["event_times"],
            "trajectory": info["trajectory_points"],
        })
        observation = observation_next
        if done:
            break
    return {
        "archetype": archetype,
        "seed": seed,
        "checkpoint": bundle["metadata"].get("generation"),
        "deliveries": records,
    }


def brain_cam(run_dir: Path, checkpoint: str, archetype: str, seed: int) -> dict:
    bundle = checkpoint_bundle(run_dir, checkpoint)
    path = _checkpoint_path(run_dir, checkpoint)
    meta = bundle["metadata"]
    return _cam_cached(
        str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size,
        archetype, int(seed), float(meta.get("noise_scale", 0.05)),
        int(meta.get("environment_version", 1)),
    )


@lru_cache(maxsize=48)
def _decision_map_cached(
    checkpoint_path: str,
    mtime_ns: int,
    size: int,
    archetype: str,
    seed: int,
    ball_index: int,
    x_index: int,
    y_index: int,
    output: str,
    resolution: int,
) -> dict:
    path = Path(checkpoint_path)
    run_dir = path.parent.parent
    checkpoint = path.stem
    bundle = _load_checkpoint_cached(str(path.resolve()), mtime_ns, size)
    policy = MLPPolicy(bundle["genome"])
    if x_index == y_index or not (0 <= x_index < policy.input_size and 0 <= y_index < policy.input_size):
        raise ValueError("choose two different valid input axes")
    if output not in {"speed", "elevation", "azimuth", "swing", "length_category"}:
        raise ValueError("unknown output")
    cam = brain_cam(run_dir, checkpoint, archetype, seed)
    if not cam["deliveries"]:
        raise ValueError("the selected over has no deliveries")
    state = np.asarray(cam["deliveries"][min(max(0, ball_index), len(cam["deliveries"]) - 1)]["observation"], dtype=float)
    xs = np.linspace(*INPUT_RANGES[x_index], resolution)
    ys = np.linspace(*INPUT_RANGES[y_index], resolution)
    xx, yy = np.meshgrid(xs, ys)
    observations = np.repeat(state.reshape(1, -1), resolution * resolution, axis=0)
    observations[:, x_index] = xx.ravel()
    observations[:, y_index] = yy.ravel()
    _, _, _, actions = _forward_batch(policy, observations)
    category_codes = {
        "full_toss": -3.0, "yorker": -2.0, "full": -1.0,
        "good": 1.0, "short": 2.0, "bouncer": 3.0,
    }
    category_labels = {str(int(value)): name.replace("_", " ") for name, value in category_codes.items()}
    output_index = OUTPUT_NAMES.index(output) if output in OUTPUT_NAMES else None
    values = actions[:, output_index] if output_index is not None else None
    if output == "speed":
        values = 115.0 + 17.5 * (values + 1.0)
    elif output == "length_category":
        pitch = PitchConditions(seam_sd_deg=0.0)
        decoded = np.empty(len(actions), dtype=float)
        for index, action in enumerate(actions):
            release = release_from_action(action, np.random.default_rng(0), noise_scale=0.0)
            trajectory = simulate_delivery(release, pitch, np.random.default_rng(index))
            decoded[index] = category_codes[trajectory.length_category]
        values = decoded
    return {
        "x_name": INPUT_NAMES[x_index], "y_name": INPUT_NAMES[y_index],
        "x_index": x_index, "y_index": y_index,
        "x_values": xs.tolist(), "y_values": ys.tolist(),
        "values": values.reshape(resolution, resolution).tolist(),
        "output": output,
        "unit": "km/h" if output == "speed" else "length category · noise-free physics" if output == "length_category" else "action [-1, 1]",
        "category_labels": category_labels if output == "length_category" else {},
        "color_center": 0.0 if output == "length_category" else 132.5 if output == "speed" else 0.0,
        "fixed_state_ball": ball_index + 1,
    }


def decision_map(
    run_dir: Path,
    checkpoint: str,
    archetype: str,
    seed: int,
    ball_index: int,
    x_index: int,
    y_index: int,
    output: str,
    resolution: int = 32,
) -> dict:
    path = _checkpoint_path(run_dir, checkpoint)
    stat = path.stat()
    return _decision_map_cached(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size, archetype,
        int(seed), int(ball_index), int(x_index), int(y_index), output, int(resolution),
    )


@lru_cache(maxsize=48)
def _brain_states_cached(
    checkpoint_path: str,
    mtime_ns: int,
    size: int,
    archetype: str,
    seed_base: int,
    target_states: int,
    environment_version: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = _load_checkpoint_cached(checkpoint_path, mtime_ns, size)
    policy = MLPPolicy(bundle["genome"])
    noise_scale = float(bundle["metadata"].get("noise_scale", 0.05))
    observations, hidden1, hidden2, outputs = [], [], [], []
    over_index = 0
    while len(observations) < target_states:
        rng = np.random.default_rng(seed_base + over_index * 101)
        env = OverEnv(noise_scale=noise_scale, environment_version=environment_version)
        observation = env.reset(archetype, rng)
        while True:
            h1, h2, raw = policy.trace(observation)
            observations.append(_adapt_observations(policy, observation).reshape(-1))
            hidden1.append(h1)
            hidden2.append(h2)
            outputs.append(np.tanh(raw))
            observation, _, done, _ = env.step(np.tanh(raw))
            if done or len(observations) >= target_states:
                break
        over_index += 1
    return (
        np.asarray(observations, dtype=float), np.asarray(hidden1, dtype=float),
        np.asarray(hidden2, dtype=float), np.asarray(outputs, dtype=float),
    )


def brain_states(
    run_dir: Path,
    checkpoint: str,
    archetype: str,
    seed: int = 9_902_101,
    target_states: int = 2_000,
):
    path = _checkpoint_path(run_dir, checkpoint)
    if not path.exists():
        raise FileNotFoundError(path.name)
    stat = path.stat()
    return _brain_states_cached(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size,
        archetype, seed + ARCHETYPE_NAMES.index(archetype) * 1_000_000,
        target_states, int(checkpoint_bundle(run_dir, checkpoint)["metadata"].get("environment_version", 1)),
    )


def permutation_importance(run_dir: Path, checkpoint: str, archetype: str) -> dict:
    states, _, _, outputs = brain_states(run_dir, checkpoint, archetype)
    bundle = checkpoint_bundle(run_dir, checkpoint)
    policy = MLPPolicy(bundle["genome"])
    base_output = outputs
    rng = np.random.default_rng(91_053 + ARCHETYPE_NAMES.index(archetype))
    scores = np.zeros((len(OUTPUT_NAMES), policy.input_size), dtype=float)
    for input_index in range(policy.input_size):
        shuffled = states.copy()
        shuffled[:, input_index] = shuffled[rng.permutation(len(shuffled)), input_index]
        _, _, _, altered = _forward_batch(policy, shuffled)
        scores[:, input_index] = np.mean((base_output - altered) ** 2, axis=0)
    return {
        "archetype": archetype,
        "state_count": int(len(states)),
        "inputs": list(INPUT_NAMES[:policy.input_size]),
        "outputs": list(OUTPUT_NAMES),
        "scores": scores.tolist(),
    }


def neuron_diagnostics(run_dir: Path, checkpoint: str, archetype: str, layer_index: int, neuron_index: int) -> dict:
    if layer_index not in (0, 1):
        raise ValueError("hidden layer index must be 0 or 1")
    states, hidden1, hidden2, outputs = brain_states(run_dir, checkpoint, archetype)
    activations = (hidden1, hidden2)[layer_index]
    if not 0 <= neuron_index < activations.shape[1]:
        raise ValueError("neuron index is out of range")
    values = activations[:, neuron_index]
    correlations = []
    named_values = [(INPUT_NAMES[i], states[:, i]) for i in range(states.shape[1])]
    named_values.extend((f"{name} output", outputs[:, i]) for i, name in enumerate(OUTPUT_NAMES))
    for name, series in named_values:
        corr = float(np.corrcoef(values, series)[0, 1]) if np.std(series) > 1e-10 else 0.0
        correlations.append({"name": name, "correlation": 0.0 if not np.isfinite(corr) else corr})
    top_indices = np.argsort(values)[-8:][::-1]
    strongest = max(correlations, key=lambda row: abs(row["correlation"]))
    label = (
        f"fires with {strongest['name']}" if strongest["correlation"] >= 0
        else f"quietens with {strongest['name']}"
    ) if abs(strongest["correlation"]) >= 0.25 else "responds to a mixed delivery state"
    return {
        "layer": layer_index + 1, "neuron": neuron_index,
        "mean_activation": float(np.mean(values)),
        "peak_activation": float(np.max(values)),
        "label": label,
        "correlations": sorted(correlations, key=lambda row: abs(row["correlation"]), reverse=True),
        "top_states": [
            {"activation": float(values[i]), "inputs": states[i].tolist(), "outputs": outputs[i].tolist()}
            for i in top_indices
        ],
    }


def _expand_genome(genome: np.ndarray) -> np.ndarray:
    policy = MLPPolicy(genome)
    w1, b1, w2, b2, w3, b3 = policy._parameters()
    if policy.input_size == LEGACY_OBSERVATION_SIZE:
        w1 = np.vstack((w1, np.zeros((OBSERVATION_SIZE - LEGACY_OBSERVATION_SIZE, w1.shape[1]))))
    return np.concatenate((w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel(), w3.ravel(), b3.ravel()))


def _saved_populations(runs_root: Path):
    genomes, labels, paths_by_run = [], [], {}
    fingerprint = []
    for run_name in POPULATION_RUNS:
        run_dir = runs_root / run_name
        paths_by_run[run_name] = []
        for path in sorted((run_dir / "populations").glob("population_*.npz")):
            stat = path.stat()
            fingerprint.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
            with np.load(path, allow_pickle=False) as data:
                population = data["population"].astype(float)
                meta = json.loads(str(data["metadata_json"].item()))
            generation = int(meta["generation"])
            paths_by_run[run_name].append((generation, path))
            genomes.extend(_expand_genome(genome) for genome in population)
            labels.extend((run_name, generation, index) for index in range(len(population)))
    return np.asarray(genomes, dtype=float), labels, paths_by_run, tuple(fingerprint)


@lru_cache(maxsize=4)
def _pca_basis_cached(fingerprint: tuple):
    genomes = []
    for path_text, _, _ in fingerprint:
        with np.load(path_text, allow_pickle=False) as data:
            population = data["population"].astype(float)
        genomes.extend(_expand_genome(genome) for genome in population)
    values = np.asarray(genomes, dtype=float)
    mean = values.mean(axis=0)
    centered = values - mean
    rng = np.random.default_rng(2468)
    rank = min(24, centered.shape[0] - 1, centered.shape[1])
    omega = rng.normal(size=(centered.shape[1], rank))
    basis, _ = np.linalg.qr(centered @ omega, mode="reduced")
    for _ in range(3):
        basis, _ = np.linalg.qr(centered @ (centered.T @ basis), mode="reduced")
    reduced = basis.T @ centered
    _, singular, right = np.linalg.svd(reduced, full_matrices=False)
    components = right[:2]
    total_variance = float(np.sum(centered * centered))
    explained = (singular[:2] ** 2 / max(1e-12, total_variance)).tolist()
    return mean, components, explained


def population_cloud(run_dir: Path, runs_root: Path, selected_generation: int) -> dict:
    all_genomes, labels, paths_by_run, fingerprint = _saved_populations(runs_root)
    if not len(all_genomes):
        return {"error": "no saved population snapshots yet"}
    mean, components, explained = _pca_basis_cached(fingerprint)
    snapshots = paths_by_run.get(run_dir.name, [])
    if not snapshots:
        return {"error": f"{run_dir.name} has no saved populations yet"}
    generation, path = min(snapshots, key=lambda item: abs(item[0] - selected_generation))
    with np.load(path, allow_pickle=False) as data:
        population = data["population"].astype(float)
        fitness = data["fitness"].astype(float) if "fitness" in data.files else None
        genome_ids = data["genome_ids"].astype(int) if "genome_ids" in data.files else None
    if fitness is None:
        plan_matrix = Path(__file__).resolve().parents[2] / "outputs" / "batter" / "plan_matrix.csv"
        baselines = load_normalization_baselines(plan_matrix)
        stat = path.stat()
        fitness = np.asarray(_estimate_population_fitness(
            str(path.resolve()), stat.st_mtime_ns, stat.st_size,
            json.dumps(baselines, sort_keys=True, separators=(",", ":")),
        ))
    coords = (_expand_population(population) - mean) @ components.T
    champion_path = run_dir / "checkpoints" / "champion.npz"
    champion_coord = None
    if champion_path.exists():
        with np.load(champion_path, allow_pickle=False) as data:
            champion = data["genome"].astype(float)
        champion_coord = (_expand_genome(champion) - mean) @ components.T
    return {
        "requested_generation": selected_generation,
        "snapshot_generation": generation,
        "points": coords.tolist(), "fitness": fitness.tolist(),
        "genome_ids": genome_ids.tolist() if genome_ids is not None else None,
        "champion": champion_coord.tolist() if champion_coord is not None else None,
        "explained_variance": explained,
    }


def _expand_population(population: np.ndarray) -> np.ndarray:
    return np.asarray([_expand_genome(genome) for genome in population], dtype=float)


@lru_cache(maxsize=16)
def _estimate_population_fitness(
    path_text: str, mtime_ns: int, size: int, baselines_json: str
) -> tuple[float, ...]:
    with np.load(path_text, allow_pickle=False) as data:
        population = data["population"].astype(float)
    baselines = json.loads(baselines_json)
    scenarios = [
        (name, 22_100_000 + i * 10_000 + over * 101)
        for i, name in enumerate(ARCHETYPE_NAMES) for over in range(2)
    ]
    return tuple(
        _evaluate_genome_payload((genome, scenarios, 0.05, baselines))["fitness"]
        for genome in population
    )


def weight_statistics(run_dir: Path, checkpoint: str, archetype: str = "slogger") -> dict:
    selected_bundle = checkpoint_bundle(run_dir, checkpoint)
    states, _, _, _ = brain_states(run_dir, checkpoint, archetype, target_states=256)
    by_generation = {}
    for folder in ("checkpoints", "best_genomes"):
        for path in (run_dir / folder).glob("gen_*.npz"):
            try:
                generation = int(path.stem.split("_")[1])
            except ValueError:
                continue
            if generation not in by_generation or folder == "best_genomes":
                by_generation[generation] = path
    checkpoint_paths = [by_generation[g] for g in sorted(by_generation)]
    records = []
    previous = None
    for path in checkpoint_paths:
        bundle = checkpoint_bundle(run_dir, path.stem)
        policy = MLPPolicy(bundle["genome"])
        weights = bundle["weights"]
        hidden1, hidden2, _, _ = _forward_batch(policy, states)
        activations = (hidden1, hidden2)
        layer_records = []
        for index, matrix in enumerate(weights):
            delta = None if previous is None else float(np.linalg.norm(matrix - previous[index]))
            saturation = float(np.mean(np.abs(activations[min(index, 1)]) > 0.95)) if index < 2 else None
            layer_records.append({
                "name": LAYER_NAMES[index],
                "mean_abs_weight": float(np.mean(np.abs(matrix))),
                "std_weight": float(np.std(matrix)),
                "saturated_hidden_share": saturation,
                "l2_from_previous": delta,
            })
        records.append({
            "generation": int(bundle["metadata"].get("generation", 0)),
            "layers": layer_records,
        })
        previous = [matrix.copy() for matrix in weights]
    return {"selected_checkpoint": selected_bundle["metadata"].get("generation"), "points": records}


@lru_cache(maxsize=8)
def _lineage_records_cached(path_text: str, mtime_ns: int, size: int):
    records = {}
    with gzip.open(path_text, "rt", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            records[int(row["child_id"])] = row
    return records


def family_tree(run_dir: Path, checkpoint: str = "champion") -> dict:
    path = run_dir / "lineage.csv.gz"
    if not path.exists():
        return {"available": False, "message": "lineage was not logged for this run"}
    checkpoint_path = _checkpoint_path(run_dir, checkpoint)
    if not checkpoint_path.exists():
        return {"available": True, "nodes": [], "message": f"checkpoint {checkpoint} is not available yet"}
    with np.load(checkpoint_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"].item()))
    champion_id = meta.get("genome_id")
    champion_gen = int(meta.get("validation_generation", meta.get("generation", 0)))
    try:
        stat = path.stat()
        records = _lineage_records_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except (OSError, EOFError, gzip.BadGzipFile):
        return {"available": True, "nodes": [], "message": "lineage log is being written; retry shortly"}
    if champion_id is None:
        candidates = [row for row in records.values() if int(row["generation"]) == champion_gen]
        if candidates:
            champion_id = int(max(candidates, key=lambda row: float(row["fitness"]))["child_id"])
    if champion_id is None or int(champion_id) not in records:
        return {"available": True, "nodes": [], "message": "champion ancestry id is not in the lineage log yet"}
    nodes = []
    edges = []
    current_id = int(champion_id)
    seen = set()
    while current_id in records and current_id not in seen:
        seen.add(current_id)
        row = records[current_id]
        node = {
            "id": current_id, "generation": int(row["generation"]),
            "fitness": float(row["fitness"]),
            "was_elite": bool(int(row["was_elite"])),
            "crossover_used": bool(int(row["crossover_used"])),
        }
        nodes.append(node)
        parent_a = int(row["parent_a"]) if row["parent_a"] else None
        parent_b = int(row["parent_b"]) if row["parent_b"] else None
        if parent_a is not None and parent_a in records:
            edges.append({"from": parent_a, "to": current_id, "kind": "primary"})
            if parent_b is not None and parent_b in records:
                donor = records[parent_b]
                nodes.append({
                    "id": parent_b, "generation": int(donor["generation"]),
                    "fitness": float(donor["fitness"]), "was_elite": False,
                    "crossover_used": bool(int(donor["crossover_used"])), "donor": True,
                })
                edges.append({"from": parent_b, "to": current_id, "kind": "crossover"})
            current_id = parent_a
        else:
            break
    dedup = {node["id"]: node for node in nodes}
    return {
        "available": True,
        "champion_id": int(champion_id),
        "champion_generation": champion_gen,
        "selected_checkpoint": checkpoint,
        "nodes": list(dedup.values()), "edges": edges,
        "message": "Primary ancestry to generation 0; crossover donors shown as side branches.",
    }

