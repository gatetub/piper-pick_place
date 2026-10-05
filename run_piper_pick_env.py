"""Sanity-check the Piper fridge-pick manager-based environment.

The manager-based equivalent of

    ./isaaclab.sh -p reach_shelf_task.py --grasp

Launches the scene (Piper arm + fridge + cube on ``lower_middle_shelf_2``)
from :mod:`piper_pick_env_cfg` and steps it with either zero actions (hold
the default pose, so you can confirm the robot/fridge/cube line up) or
random actions (sanity-check that the observation/action managers and
rewards/terminations are wired correctly before training).

Usage
-----
    ./isaaclab.sh -p scripts/biolab/run_piper_pick_env.py --num_envs 4
    ./isaaclab.sh -p scripts/biolab/run_piper_pick_env.py --num_envs 4 --random_actions
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the Piper fridge-pick manager-based environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument(
    "--random_actions", action="store_true", help="Sample random actions instead of holding the default pose."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv

from piper_pick_env_cfg import PiperFridgePickEnvCfg_PLAY


def main():
    env_cfg = PiperFridgePickEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            if args_cli.random_actions:
                actions = torch.rand(env.action_manager.action.shape, device=env.device) * 2.0 - 1.0
            else:
                actions = torch.zeros(env.action_manager.action.shape, device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            cube_pos_w = env.scene["object"].data.root_pos_w[0].tolist()
            print(f"[Env 0]: reward {rew[0].item(): .3f}  cube_pos_w {[round(v, 3) for v in cube_pos_w]}")
            count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
