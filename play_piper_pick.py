"""Play back a trained Piper fridge-pick policy.

Self-contained equivalent of ``scripts/reinforcement_learning/rsl_rl/play.py``
for the out-of-tree task in this folder.

Usage
-----
    ./isaaclab.sh -p scripts/biolab/play_piper_pick.py --num_envs 16
    ./isaaclab.sh -p scripts/biolab/play_piper_pick.py --checkpoint logs/rsl_rl/piper_fridge_pick/.../model_1500.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Piper fridge-pick policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific model checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import get_checkpoint_path

from piper_pick_agent_cfg import agent_cfg
from piper_pick_env_cfg import PiperFridgePickEnvCfg_PLAY

INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def main():
    env_cfg = PiperFridgePickEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent = agent_cfg()
    agent = handle_deprecated_rsl_rl_cfg(agent, INSTALLED_RSL_RL_VERSION)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent.experiment_name))

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent.load_run, agent.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)

    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
