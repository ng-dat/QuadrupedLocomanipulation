import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo

from environments.box_env import BoxEnvironment


def train(steps=1000000):
    env = BoxEnvironment()

    NUM_CPUS = 4
    NUM_SAMPLES_EACH_WORKER = int(4096 / NUM_CPUS)
    config = ppo.DEFAULT_CONFIG.copy()
    config.update({
        "env": env,  # DummyQuadrupedEnv,
        "num_gpus": 0,
        "num_workers": NUM_CPUS,
        "num_envs_per_worker": 1,  # to test if same as SB
        "lr": 1e-4,
        "monitor": True,
        "model": {"use_lstm": True},
        "train_batch_size": NUM_CPUS * NUM_SAMPLES_EACH_WORKER,  # 4096, #n_steps,
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 128,  # try btw 64, 128, 256
        "rollout_fragment_length": NUM_SAMPLES_EACH_WORKER,
        "clip_param": 0.2,
        "vf_clip_param": 1,  # try btw 0.2,1, 10
        "vf_loss_coeff": 0.5,
        "lambda": 0.95,
        "grad_clip": 0.5,
        "kl_coeff": 0.0,
        "kl_target": 0.01,
        "entropy_coeff": 0.0,
        "observation_filter": "MeanStdFilter",  # THIS IS NEEDED IF NOT USING VEC_NORMALIZE
        "clip_actions": True,
        "vf_share_layers": False,
        "normalize_actions": True,
        "preprocessor_pref": "rllib",  # what are these even doing? anything?
        "simple_optimizer": False,  # test this next
        "batch_mode": "truncate_episodes",
        "no_done_at_end": False,
        "shuffle_buffer_size": NUM_CPUS * NUM_SAMPLES_EACH_WORKER,  # 4096,
    })

    tune.run(
        "PPO",
        config=config,
        stop={"episode_reward_mean": 2, "timesteps_total": 100000000},
        local_dir="./logs/results/",
        checkpoint_freq=20,
        verbose=2,
    )


if __name__ == "__main__":
    ray.init()
    # train()
    tune.run(
        "PPO",
        stop={"episode_reward_mean": 200},
        config={
            "env": "CartPole-v0",
            "num_gpus": 0,
            "num_workers": 1,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        },
    )
    ray.shutdown()