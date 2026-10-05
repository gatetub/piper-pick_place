"""RSL-RL PPO hyperparameters for the Piper fridge-pick task.

Split out from train_piper_pick.py (and reused by play_piper_pick.py) so that
importing the agent config doesn't also re-run train_piper_pick's argparse /
AppLauncher boilerplate.
"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO hyperparameters -- the same ones Isaac Lab ships for Franka cube-lift."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1500,
        save_interval=50,
        experiment_name="piper_fridge_pick",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.006,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
