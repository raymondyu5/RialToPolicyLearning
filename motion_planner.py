# cuRobo
import torch
import numpy as np
from curobo.geom.types import Sphere, Cuboid
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
    CudaRobotModelConfig,
)
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


class MotionPlanner:

    def __init__(
        self,
        device=None,
        # TODO: Unlikely these configs apply to RialTo
        robot_file="franka.yml",
        world_file="collision_table.yml",
        interpolation_dt=0.1,
        random_obstacle=False,
    ):
        self.device = device
        self.tensor_args = TensorDeviceType(device=device)
        self.interpolation_dt = interpolation_dt

        # create configs
        robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"],
            self.tensor_args,
        )
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        # TODO: This is probably not correct for our purposes...
        
        # leads to IK failure, not sure why
        # add random obstacle to world -> more diverse trajectories
        if random_obstacle:
            pos = np.random.uniform([0.3, -0.2, 0.2], [0.7, 0.2, 0.3])
            # obstacle_0 = Sphere(
            #     name="obstacle_0", radius=2.6, pose=[*pos, 0.0, 0.0, 0.0, 0.0]
            # )
            obstacle_0 = Cuboid(
                # name="obstacle_0", dims=[0.1, 0.1, 0.1], pose=[*pos, 0.0, 0.0, 0.0, 0.0]
                name="obstacle_0", dims=[0.1, 0.1, 0.1], pose=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            )
            world_cfg.add_obstacle(obstacle_0)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)

        # MP
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            self.tensor_args,
            interpolation_dt=self.interpolation_dt,
            # # Zoey params
            # trajopt_tsteps=50,
            # interpolation_steps=10000,
            # rotation_threshold=0.01,
            # position_threshold=0.001,
            # num_ik_seeds=100,
            # num_trajopt_seeds=50,
            # collision_checker_type=CollisionCheckerType.PRIMITIVE,
            # grad_trajopt_iters=500,
            # trajopt_dt=0.5,
            # evaluate_interpolated_trajectory=True,
            # js_trajopt_dt=0.5,
            # js_trajopt_tsteps=34,
            # velocity_scale=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5]
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)

        # FK
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        # IK
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=False, # True,
            self_collision_opt=False, # True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)

        self.retract_cfg = self.motion_gen.get_retract_config()

    def plan_motion(self, ee_pose, target_ee_pose, return_ee_pose=False):

        start = Pose(
            self.tensor_args.to_device([ee_pose[:3]]),
            self.tensor_args.to_device([ee_pose[3:]]),
            normalize_rotation=False,
        )

        result = self.ik_solver.solve_single(start)
        qpos = result.solution[0]
        start = JointState.from_position(
            qpos,
            self.kin_model.joint_names,
            # self.tensor_args.to_device([qpos]), self.kin_model.joint_names
        )

        goal = Pose(
            self.tensor_args.to_device([target_ee_pose[:3]]),
            self.tensor_args.to_device([target_ee_pose[3:]]),
            normalize_rotation=False,
        )

        result = self.motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(max_attempts=1)
        )

        traj = result.get_interpolated_plan()
        if traj is None:
            raise ValueError
        # print(
        #     f"Trajectory Generated: success {result.success.item()} | len {len(traj)} | optimized_dt {result.optimized_dt.item()}"
        # )
        
        # replace joint position with ee pose
        if return_ee_pose:
            traj = self.kin_model.get_state(traj.position)

        return traj