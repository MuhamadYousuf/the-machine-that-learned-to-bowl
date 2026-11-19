import numpy as np

from aibowler.env import MAX_LEGAL_BALLS, OverEnv
from aibowler.evolve import run_toy_ga
from aibowler.outcome import BallOutcome
from aibowler.policy import GENOME_SIZE, MLPPolicy


def safe_action():
    return np.array([0.0, -0.05, 0.0, 0.0])


def test_over_ends_after_six_legal_balls_and_observation_is_normalized():
    env = OverEnv(noise_scale=0.0)
    observation = env.reset("technician", np.random.default_rng(12))
    assert observation.shape == (18,)
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0) and np.all(observation <= 1.0)

    steps = 0
    done = False
    while not done and steps < 12:
        observation, _, done, _ = env.step(safe_action())
        assert observation.shape == (18,)
        assert np.all(observation >= -1.0) and np.all(observation <= 1.0)
        steps += 1
    assert done
    assert env.legal_balls == MAX_LEGAL_BALLS
    assert env.illegal_balls <= 6


def test_illegal_balls_are_rebowled_and_sixth_extra_ends_over_with_penalty():
    env = OverEnv(noise_scale=0.0)
    env.reset("slogger", np.random.default_rng(13))
    beamer_action = np.array([0.0, 1.0, 0.0, 0.0])
    done = False
    final_reward = 0.0
    while not done:
        _, final_reward, done, info = env.step(beamer_action)
        assert info["outcome"].result == "no_ball"
        assert env.legal_balls == 0
    assert env.illegal_balls == 6
    assert info["extra_penalty_applied"]
    assert final_reward <= -10.0


def test_wicket_brings_a_fresh_batter_and_resets_delivery_history(monkeypatch):
    env = OverEnv(noise_scale=0.0)
    env.reset("slogger", np.random.default_rng(14))
    old_state = env.state
    old_batter = env.current_archetype

    def force_wicket(features, archetype, state, rng):
        return BallOutcome(
            result="bowled", runs_conceded=0, runs_scored=0,
            is_wicket=True, is_legal_ball=True, shot="miss",
            shot_angle_deg=None, shot_distance_m=None,
            probabilities={"bowled": 1.0}, expected_runs=0.0, wicket_prob=1.0,
            surprise=0.0, miss_probability=1.0, edge_probability=0.0,
            connect_probability=0.0, batting_run_probabilities={"dot": 1.0},
        )

    monkeypatch.setattr("aibowler.env.resolve", force_wicket)
    _, _, _, info = env.step(safe_action())
    assert info["batter_archetype"] == old_batter.name
    assert env.state is not old_state
    assert env.current_archetype is not old_batter
    assert env.state.balls_faced == 0
    assert len(env.history) == 0


def test_policy_genome_round_trip_is_exact():
    rng = np.random.default_rng(15)
    policy = MLPPolicy.random(rng)
    genome = policy.get_genome()
    assert genome.shape == (GENOME_SIZE,)
    replacement = np.linspace(-1.0, 1.0, GENOME_SIZE)
    policy.set_genome(replacement)
    assert np.array_equal(policy.get_genome(), replacement)
    obs = np.zeros(17)
    assert policy.act(obs).shape == (4,)
    assert np.all(policy.act(obs) >= -1.0) and np.all(policy.act(obs) <= 1.0)


def test_legacy_policy_and_environment_schema_still_round_trip():
    from aibowler.env import LEGACY_OBSERVATION_SIZE
    from aibowler.policy import LEGACY_GENOME_SIZE

    legacy = MLPPolicy(np.zeros(LEGACY_GENOME_SIZE))
    env = OverEnv(noise_scale=0.0, environment_version=1)
    observation = env.reset("slogger", np.random.default_rng(150))
    assert observation.shape == (LEGACY_OBSERVATION_SIZE,)
    assert legacy.input_size == LEGACY_OBSERVATION_SIZE
    assert legacy.act(observation).shape == (4,)


def test_visual_model_interpolation_handles_weight_shape_growth():
    from aibowler.viz.render import interpolate_models

    earlier = {"weights": [[[0.0, 2.0]]], "biases": [[-1.0]]}
    later = {"weights": [[[2.0, 4.0], [6.0, 8.0]]], "biases": [[1.0, 3.0]]}
    middle = interpolate_models(earlier, later, 0.5)
    assert middle["weights"] == [[[1.0, 3.0], [6.0, 8.0]]]
    assert middle["biases"] == [[0.0, 3.0]]
    assert interpolate_models(earlier, later, 0.0)["weights"] == [[[0.0, 2.0], [6.0, 8.0]]]
    assert interpolate_models(earlier, later, 1.0)["weights"] == later["weights"]


def test_bouncer_quota_makes_third_above_shoulder_delivery_a_one_run_no_ball():
    env = OverEnv(noise_scale=0.0)
    env.reset("slogger", np.random.default_rng(151))
    bouncer_action = np.array([0.43, -0.89, 0.138, 0.0])
    records = [env.step(bouncer_action)[3] for _ in range(3)]
    assert [record["bouncers_this_over"] for record in records] == [1, 2, 3]
    assert not records[0]["bouncer_no_ball"]
    assert not records[1]["bouncer_no_ball"]
    assert records[2]["bouncer_no_ball"]
    third = records[2]["outcome"]
    assert third.result == "no_ball"
    assert third.runs_conceded == 1
    assert third.runs_scored == 0
    assert not third.is_legal_ball
    assert not third.is_wicket
    assert third.expected_runs == 1.0
    assert env.legal_balls == 2
    assert env.extras == 1


def test_ga_improves_on_a_tiny_quadratic_toy_problem():
    history = run_toy_ga(
        lambda genome: -float(np.sum((genome - 0.25) ** 2)),
        genome_size=6,
        population_size=36,
        generations=30,
        seed=16,
    )
    assert history[-1] > history[0]


def test_environment_is_deterministic_with_fixed_seed():
    outputs = []
    for _ in range(2):
        env = OverEnv(noise_scale=0.05)
        observation = env.reset("tailender", np.random.default_rng(17))
        records = []
        done = False
        while not done:
            observation, reward, done, info = env.step(safe_action())
            records.append((reward, info["action"], info["outcome"].result))
        outputs.append(records)
    assert outputs[0] == outputs[1]


def test_environment_reuses_common_batter_and_execution_noise_per_delivery():
    first_env = OverEnv(noise_scale=0.05)
    second_env = OverEnv(noise_scale=0.05)
    first_env.reset("technician", np.random.default_rng(18))
    second_env.reset("technician", np.random.default_rng(18))
    assert first_env.current_archetype == second_env.current_archetype

    first_requested = np.array([0.10, -0.10, 0.10, -0.10])
    second_requested = np.array([-0.10, 0.10, -0.10, 0.10])
    _, _, _, first_info = first_env.step(first_requested)
    _, _, _, second_info = second_env.step(second_requested)
    first_noise = np.asarray(first_info["action"]) - first_requested
    second_noise = np.asarray(second_info["action"]) - second_requested
    assert np.allclose(first_noise, second_noise)

