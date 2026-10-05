"""Manager-based RL environment: AgileX Piper arm learns to pick a cube off a
fridge shelf.

This replaces the scripted waypoint/IK pipeline in
``fridge_manipulation/fridge_tasks/scripts/reachtask/reach_shelf_task.py``
(run with ``--grasp``) with an ``isaaclab.envs.ManagerBasedRLEnv``: instead of
a hand-written state machine driving the arm through fixed waypoints, the
robot gets an observation vector and a reward and has to learn the
reach-grasp-lift behaviour itself. The arm's action is still Cartesian/IK
(``DifferentialInverseKinematicsActionCfg``, matching the base-frame-correct
convention validated in ``comeout4.py``) -- the policy commands end-effector
pose deltas and Isaac Lab's IK action term solves the joint targets, rather
than the policy having to learn forward kinematics from scratch. What is
learned is *when and where* to move the end-effector, not the waypoint
sequence itself.

The MDP (actions, observations, reward terms, reset events, terminations) is
reused as-is from Isaac Lab's own Franka "Cube Lift" task
(``isaaclab_tasks.manager_based.manipulation.lift``) -- the manager-based
counterpart of the ``franka/pick_place.py`` standalone demo. That reward/obs
design already solves "reach object -> close gripper -> lift -> carry to
goal" for a 7-DOF arm + parallel gripper, which is exactly this task's shape
for the Piper's 6-DOF arm + parallel gripper, so it is imported rather than
reimplemented.

Only the scene is new: the Franka + table + loose cube is swapped for the
Piper arm + the fridge asset (``fridge4.usd``) + a cube placed inside
``lower_middle_shelf_2``, using the exact measured offsets from
``fridge_geometry.py`` (robot placed at the scene/env origin, fridge and cube
positioned relative to it).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

# Reuse Isaac Lab's own Franka cube-lift MDP (reach/lift/goal-tracking rewards,
# the joint_pos/joint_vel/object-position observations, the reset events, the
# drop/time-out terminations). Nothing in it is Franka-specific.
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

# ---------------------------------------------------------------------------
# Asset paths -- the same USD files the scripted pipeline uses, unmodified.
# ---------------------------------------------------------------------------
_ASSET_DIR = "/media/avishikta/data/Avishikta/fridge_manipulation/fridge_tasks/assets"
FRIDGE_USD = f"{_ASSET_DIR}/fridge4.usd"
PIPER_USD = f"{_ASSET_DIR}/piper_with_gripper.usd"

# ---------------------------------------------------------------------------
# Scene geometry. The scripted script places the robot at world
# (0.237, 0.541, 0.0) and the fridge/cube relative to that; here the robot is
# shifted to the env origin and the fridge/cube offsets carried over as a
# rigid translation (same relative layout, simpler numbers).
#   fridge = fridge_translate_world - robot_pos_world = (-0.42, 0.12, 0.0)
#   cube   = cube_pos_world         - robot_pos_world = (-0.40, 0.00, 0.32)
# Both the fridge and the robot are yawed 180 deg about Z in the source
# scene, which is why the quaternion below is (0, 0, 0, 1) and not identity.
# ---------------------------------------------------------------------------
_YAW_180 = (0.0, 0.0, 0.0, 1.0)
ROBOT_POS = (0.0, 0.0, 0.0)
FRIDGE_POS = (-0.42, 0.12, 0.0)
CUBE_POS = (-0.40, 0.0, 0.32)  # sits on lower_middle_shelf_2 (surface_z = 0.3038)

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_EE_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


PIPER_ARM_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS,
        rot=_YAW_180,
        # Neutral "ready" pose outside the fridge, identical to comeout4.py's
        # PIPER_ARM_CFG -- the arm starts clear of the fridge opening and has
        # to reach IN to the cube itself, like the real scripted task, rather
        # than starting already inside next to the cube (which doesn't
        # generalize across the randomized cube reset range and isn't
        # representative of the actual task).
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.3,
            "joint3": -0.5,
            "joint4": 0.0,
            "joint5": 0.3,
            "joint6": 0.0,
            # gripper starts OPEN (ready to grasp) rather than closed. The two
            # gripper joints are mirrored (joint2's range is negative), so a
            # shared regex value can't be used.
            "gripper_joint1": 0.04,
            "gripper_joint2": -0.04,
        },
    ),
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit=40.0,
            velocity_limit=3.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper.*"],
            effort_limit=20.0,
            velocity_limit=0.2,
            stiffness=2000.0,
            damping=100.0,
        ),
    },
)


##
# Scene definition
##


@configclass
class PiperFridgeSceneCfg(InteractiveSceneCfg):
    """Piper arm + fridge + cube-on-a-shelf scene."""

    # Floor/table the robot and fridge actually stand on. fridge_geometry.py's
    # measurements (ROBOT_POS_W z=0, FRIDGE_TRANSLATE z=0) are made in
    # scene_1copy.usd's frame, where z=0 IS the floor the robot is mounted to
    # -- we just never modeled that surface here, which is why the arm looked
    # like it was floating in mid-air. A ground plane at z=-0.2 (the previous
    # value) put empty space under the robot's base and let a dropped cube
    # fall 0.2 m+ away before landing, out of reach for the rest of the
    # episode. Flush at z=0 it is both the visual mount and a catch surface
    # close enough that a dropped cube stays near the arm.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    fridge = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fridge",
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRIDGE_POS, rot=_YAW_180),
        spawn=sim_utils.UsdFileCfg(
            usd_path=FRIDGE_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    robot: ArticulationCfg = PIPER_ARM_CFG

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS, rot=_YAW_180),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=1.0,
                angular_damping=1.0,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.2, restitution=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
    )

    # end-effector frame, same body the scripted IK pipeline uses ("link6")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link6",
        debug_vis=False,
        visualizer_cfg=_EE_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link6",
                name="end_effector",
                # TCP offset along link6's forward axis, same value comeout4/piper_reach use.
                offset=OffsetCfg(pos=(0.0, 0.0, 0.08)),
            ),
        ],
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Target pose for the cube: lift it off the shelf and bring it toward the front of the fridge."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # robot-local frame; the robot is yawed 180 deg about Z, so a
            # local (+x, +y, +z) offset lands at world (-x, -y, +z) -- i.e.
            # roughly above and in front of the cube's shelf position.
            pos_x=(0.30, 0.45),
            pos_y=(-0.05, 0.05),
            pos_z=(0.40, 0.50),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Cartesian (differential IK) control for the arm, binary open/close for the gripper.

    Matches the kinematics convention validated in the scripted
    ``comeout4.py`` pipeline (base-frame Jacobian via a relative-pose
    DifferentialIKController, TCP 0.08 m forward of ``link6``) rather than
    commanding raw joint angles -- the policy outputs end-effector pose
    deltas and Isaac Lab's own (already base-frame-correct)
    ``DifferentialInverseKinematicsAction`` solves the joint targets, instead
    of the policy having to learn forward kinematics from scratch. ``scale``
    is kept small (0.05 m / step) for the same reason the old joint-space
    scale was lowered: the fridge compartment is tight and a 0.1 kg cube on a
    narrow shelf gets knocked off by anything more violent during early,
    untrained/random exploration.
    """

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        body_name="link6",
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.08)),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.05,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_joint1", "gripper_joint2"],
        # joint2 is mirrored (negative range), so open/close is +-0.04, not a shared value.
        open_command_expr={"gripper_joint1": 0.04, "gripper_joint2": -0.04},
        close_command_expr={"gripper_joint1": 0.0, "gripper_joint2": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observations for the policy: joint state, cube position, goal, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: full scene reset, plus a small randomization of the cube's position on the shelf."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # stays within lower_middle_shelf_2's measured bay (0.24 x 0.20 m)
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reach -> lift -> carry-to-goal, identical weighting to the Franka cube-lift task."""

    # std widened from the Franka task's 0.1 -- gives a non-trivial reward
    # gradient even while the end-effector is still far from the cube, instead
    # of ~0 reward until it gets lucky enough to land within 10 cm.
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.25}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.38}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.38, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.38, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """End the episode on time-out, or if the cube is knocked off the shelf."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Immediate, like the Franka cube-lift task. A warmup delay was tried to
    # give early episodes more learning signal, but measurement showed it
    # backfires: the cube falls clear out of the fridge and settles far away
    # (confirmed via a step-by-step trace), so the episode runs its full
    # length with ~68% dead, zero-reward steps once the cube is unreachable --
    # diluting reaching_object back to ~0 instead of helping. Ending the
    # episode as soon as the cube is lost gives a short, clean episode
    # instead of a long diluted one.
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.1, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class PiperFridgePickEnvCfg(ManagerBasedRLEnvCfg):
    """Piper arm learns to pick a cube off a fridge shelf."""

    scene: PiperFridgeSceneCfg = PiperFridgeSceneCfg(num_envs=512, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class PiperFridgePickEnvCfg_PLAY(PiperFridgePickEnvCfg):
    """Small, deterministic variant for visual sanity-checks / policy playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
