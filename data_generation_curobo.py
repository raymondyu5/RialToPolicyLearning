import time
import argparse
import wandb
import numpy as np
import torch
import yaml
import os
from utils import create_env, preprocess_pcd_from_canon, create_panda_urdf
from motion_planner import MotionPlanner
from transforms3d.euler import euler2quat, quat2euler
from os import listdir
from os.path import isfile, join
import logging
logging.getLogger('curobo').setLevel(logging.WARNING)

def run(**cfg):
    if cfg["from_disk"]:
        cfg["render_images"] = False
    env_name = cfg["env_name"]
    demo_folder = cfg["demo_folder"]
    num_demos = cfg["num_demos"]
    max_path_length = cfg["max_path_length"]
    offset = cfg["offset"]
    save_all = cfg["save_all"]

    env, _ = create_env(cfg, cfg['display'], seed=cfg['seed'])
    os.makedirs(f"demos/{env_name + demo_folder}", exist_ok=True)

    collect_demos(env, num_demos, env_name, max_path_length, offset, save_all, demo_folder, cfg=cfg)

class CartesianPDController:
    def __init__(self, Kp, Kd):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.pos_prev_error = 0
        self.quat_prev_error = 0
        # RialTo doesn't seem to use hz other than in real
        # self.dt = 1 / control_hz
        self.dt = 1

    def reset(self):
        self.pos_prev_error = 0
        self.quat_prev_error = 0

    def update(self, curr, des):
        # Calc the position error
        pos_error = des[:3] - curr[:3]
        # derivative of the position error
        pos_error_dot = (pos_error - self.pos_prev_error) / self.dt
        # update previous position error and time for the next iteration
        self.pos_prev_error = pos_error
        # Calc the position control output
        u_pos = self.Kp * pos_error + self.Kd * pos_error_dot

        # Calc the quaternion error
        quat_error = des[3:] - curr[3:]
        quat_error = np.arctan2(np.sin(quat_error), np.cos(quat_error))
        # Calc the derivative of the quaternion error
        quat_error_dot = (quat_error - self.quat_prev_error) / self.dt
        # Update the previous quaternion error and time for the next iteration
        self.quat_prev_error = quat_error
        # Calc the quaternion control output
        u_quat = self.Kp * quat_error + self.Kd * quat_error_dot

        u = np.concatenate((u_pos, u_quat))
        return u

# TODO: Definitely needs tuning.
def move_to_cartesian_pose(target_pose, gripper, motion_planner, controller, env, urdf, cfg, progress_threshold=1e-3, max_iter_per_waypoint=20):
    controller.reset()
    start = env.unwrapped._robot.get_ee_pose().copy()
    start = np.concatenate((start[:3], euler2quat(start[3:])))
    target_pose = target_pose.copy()
    # between -pi / 2 and pi / 2
    if target_pose[5] > np.pi / 2:
        target_pose[5] -= np.pi
    if target_pose[5] < -np.pi / 2:
        target_pose[5] += np.pi

    goal = np.concatenate((target_pose[:3], euler2quat(target_pose[3:])))
    # goal = env.extract_goal(env.sample_goal())
    qpos_plan = motion_planner.plan_motion(start, goal, return_ee_pose=True)

    # TODO: Need a second eye on this: 
    joint = env.base_env._env.get_robot_joints()
    pcd = env.render_image(sensors=["distance_to_image_plane"])[1]

    pcd_processed_points = preprocess_pcd_from_canon(pcd, joint, urdf, urdf.canonical_link_pcds, cfg)
    steps = 0
    actions = []
    states = []
    joints = []
    points = []
    imgs = []
    # first waypoint is current pose -> start from i=1
    for i in range(len(qpos_plan.ee_position) - 1):
        des_pose = np.concatenate((qpos_plan.ee_position[i + 1].cpu().numpy(), quat2euler(qpos_plan.ee_quaternion[i].cpu().numpy())))
        last_curr_pose = env.unwrapped._robot.get_ee_pose()
        # In this case here, waypoints are given from the motion planner
        for j in range(max_iter_per_waypoint):
            curr_pose = env.unwrapped._robot.get_ee_pose()
            # run PD controller
            act = controller.update(curr_pose, des_pose)
            act = np.concatenate((act, [gripper]))

            # step env
            obs, _, done, info = env.step(act)
            steps += 1

            actions.append(act)
            states.append(obs)
            joints.append(info["robot_joints"].detach().cpu().numpy())
            pcd = env.render_image(sensors=["distance_to_image_plane"])[1]
            pcd_processed_points = preprocess_pcd_from_canon(pcd, info["robot_joints"], urdf, urdf.canonical_link_pcds, cfg)
            points.append(pcd_processed_points[0])
            imgs.append(env.render_image(sensors=["rgb"])[0][0])

            curr_pose = env.unwrapped._robot.get_ee_pose()
            pos_diff = curr_pose[:3] - last_curr_pose[:3]
            angle_diff = curr_pose[3:] - last_curr_pose[3:]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            err = np.linalg.norm(pos_diff) + np.linalg.norm(angle_diff)

            # early stopping if actions don't change pos anymore
            if err < progress_threshold:
                break

            last_curr_pose = curr_pose

        # if done:
        #     break

    return actions, states, joints, points, imgs, steps

def collect_demos(env, num_demos, env_name, max_path_length, offset=0, save_all=False, demo_folder="", cfg=None):
    i = offset
    urdf = create_panda_urdf(cfg)
    print(env_name + demo_folder)
    # TODO: Is any of this disk stuff needed? The randomizing traj looks somewhat important
    # if cfg["from_disk"]:
    #     num_demos_disk = get_num_demos(cfg)

    motion_planner = MotionPlanner(interpolation_dt=0.1, random_obstacle=False, device=torch.device("cuda:0"))
    controller = CartesianPDController(Kp=1.0, Kd=0.0, control_hz=env.unwrapped._robot.control_hz)

    while i < num_demos + offset:
        actions = []
        states = []
        points = []
        joints = []
        video = []
        
        env.reset()
        # pcd = env.render_image(sensors=["distance_to_image_plane"])[1]
        # pcd_processed_points = preprocess_pcd_from_canon(pcd, joint, urdf, urdf.canonical_link_pcds, cfg)
        print(env.action_space)
        print("Max path length", max_path_length)
        t = 0
        # This is from a obj wrapper in mujoco, need a replacement most likely
        # https://github.com/memmelma/polymetis_franka/blob/d0f468e0f0e2dce31e25b3d3154a708c1ad6c01c/robot/sim/mujoco/obj_wrapper.py#L56
        target_pos = env.get_obj_pose().copy()[:3] #TODO: Unsure if env.get_obj_pose() works, otherwise use: get_object_pos, also get_object_joints and get_endeff_pos exist
        target_quat = quat2euler(env.get_obj_pose().copy()[3:])
        target_pose = np.concatenate((target_pos, target_quat))
        target_pose[3:5] = env.unwrapped._default_angle[:2]
        # TODO: z val for curobo, unsure if needed. Was probably part of real in OpenRT
        target_pose[2] = 0.12

        # Move to goal
        action_seq, obs_seq, joint_seq, point_seq, image_seq, steps = move_to_cartesian_pose(target_pose, 0.0, motion_planner, controller, env, urdf, cfg)
        t += steps
        actions.extend(action_seq)
        states.extend(obs_seq)
        points.extend(point_seq)
        joints.extend(joint_seq)
        video.extend(image_seq)

        target_pose[2] = 0.2
        # Pick up, gripper = 1.0 is to close the gripper
        action_seq, obs_seq, joint_seq, point_seq, image_seq, steps = move_to_cartesian_pose(target_pose, 1.0, motion_planner, controller, env, urdf, cfg)
        t += steps
        actions.extend(action_seq)
        states.extend(obs_seq)
        points.extend(point_seq)
        joints.extend(joint_seq)
        video.extend(image_seq)
        #TODO: Unsure what this success condition actually does, probably specified in sim
        success = env.base_env._env.get_success().detach().cpu().numpy()
        
        if success and t <= max_path_length:
            actions = np.array([actions])
            states = np.array([states])
            points = np.array([points])
            joints = np.array([joints])

            print("Saving in", env_name + demo_folder)
            np.save(f"demos/{env_name + demo_folder}/actions_0_{i}.npy", actions)
            np.save(f"demos/{env_name + demo_folder}/states_0_{i}.npy", states)
            np.save(f"demos/{env_name + demo_folder}/joints_0_{i}.npy", joints)
            np.save(f"demos/{env_name + demo_folder}/pcd_points_0_{i}.npy", points)
            if not cfg["from_disk"] and len(video):  # todo(as) - add a video recording for the carb version as well
                create_video(video, f"{env_name}")
            i += 1

def create_video(images, video_filename):
    images = np.array(images).astype(np.uint8)
    images = images.transpose(0, 3, 1, 2)
    wandb.log({"demos_video_trajectories": wandb.Video(images, fps=10)})

def get_data_foldername(cfg):
    filename = cfg["datafilename"]
    if "datafolder" in cfg:
        datafolder = cfg["datafolder"]
    else:
        datafolder = "/data/pulkitag/misc/data/"
    folder_name = f"{datafolder}/{filename}"
    return folder_name

def get_num_demos(cfg):
    folder_name = get_data_foldername(cfg)
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    num_demos_disk = len(onlyfiles)
    return num_demos_disk // 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--epsilon_greedy_rollout", type=float, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--usd_name", type=str, default=None)
    parser.add_argument("--usd_path", type=str, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--demo_folder", type=str, default="")
    parser.add_argument("--not_randomize", action="store_true", default=False)
    parser.add_argument("--from_disk", action="store_true", default=False)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument("--datafilename", type=str, default=None)
    parser.add_argument("--distractors", type=str, default="no_distractors")
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--camera_params", type=str, default="teleop_toynbowl")

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]
    params.update(config[args.env_name])
    params.update({'randomize_pos': not args.not_randomize, 'randomize_rot': not args.not_randomize})

    if args.extra_params is not None:
        all_extra_params = args.extra_params.split(",")
        for extra_param in all_extra_params:
            params.update(config[extra_param])

    data_folder_name = f"{args.env_name}_teleop_{args.seed}"
    params.update(config["teleop_params"])

    params.update(config[args.camera_params])
    params.update(config[args.distractors])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value

    params["data_folder"] = data_folder_name

    if "WANDB_DIR" in os.environ:
        wandb_dir = os.environ["WANDB_DIR"]
    else:
        wandb_dir = params["wandb_dir"]

    # wandb.init(project=args.env_name+"teleop_demos", name=f"{args.env_name}_demos", config=params, dir=wandb_dir)
    wandb.init(mode="disabled")
    run(**params)
