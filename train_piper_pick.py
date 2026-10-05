"""Train the Piper fridge-pick policy with RSL-RL (PPO).

Self-contained equivalent of ``scripts/reinforcement_learning/rsl_rl/train.py``:
builds :class:`piper_pick_env_cfg.PiperFridgePickEnvCfg` directly instead of
going through the gym task registry, since this task lives entirely under
``scripts/biolab`` rather than inside the ``isaaclab_tasks`` package.

Usage
-----
    ./isaaclab.sh -p scripts/biolab/train_piper_pick.py --num_envs 512 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Piper fridge-pick task with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from the last checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from piper_pick_agent_cfg import agent_cfg
from piper_pick_env_cfg import PiperFridgePickEnvCfg
from isaaclab_tasks.utils import get_checkpoint_path

INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def main():
    env_cfg = PiperFridgePickEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent = agent_cfg()
    if args_cli.max_iterations is not None:
        agent.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent.seed = args_cli.seed
    env_cfg.seed = agent.seed
    # migrate the policy/algorithm cfg to whatever schema the installed rsl-rl-lib expects
    agent = handle_deprecated_rsl_rl_cfg(agent, INSTALLED_RSL_RL_VERSION)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent.run_name:
        log_dir += f"_{agent.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)

    if args_cli.resume:
        resume_path = get_checkpoint_path(log_root_path, agent.load_run, agent.load_checkpoint)

    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
    runner.add_git_repo_to_log(__file__)
    if args_cli.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent)
    print_dict({"num_envs": env_cfg.scene.num_envs, "device": env_cfg.sim.device}, nesting=1)

    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
