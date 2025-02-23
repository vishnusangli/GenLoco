# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple locomotion taskand termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import os
import numpy as np

from motion_imitation.envs.env_wrappers import imitation_terminal_conditions
from motion_imitation.utilities import pose3d
from motion_imitation.utilities import motion_data
from motion_imitation.utilities import motion_util
from pybullet_utils import transformations

#velocity, energy, pose, height, deviation, joint angle
SUBREWARD_WEIGHTS = np.array([0.5, 0.15, 0.1, 0.0, 0.0, 0.15])

### Loco ### (Tolerance)
TARGET_VELOCITY = 0.8
VEL_UPPER_BOUND = 1.5*TARGET_VELOCITY
LOCO_SLOPE = 1.0

### Energy ### (exp)
ENERGY_EXP_SCALE = 1e-2

### POSE ### (exp)
POSE_SCALING=3

### HEIGHT ### (Tolerance)
WALKING_MIN_HEIGHT=0.25
HEIGHT_SLOPE = 0.5

### DEVIATION ### (exp)
DEVIATION_SCALING=30

### JOINT ANGLES ### 
ANGLE_SCALING=3e-1


def linear_sigmoid(x, val_at_1):
    scale = 1 - val_at_1
    scaled_x = x *scale
    return np.where(np.abs(scaled_x) < 1, 1 - scaled_x, 0.)

def my_tolerance(x, lower_bound, upper_bound, margin, value_at_margin):
    #no checking done for values, double check the inputs
    in_bounds = np.logical_and(lower_bound <= x, x <= upper_bound)
    if np.nonzero(margin):
        d = np.where(x < lower_bound, lower_bound - x, x - upper_bound) / margin
        value = np.where(in_bounds, 1., linear_sigmoid(d, value_at_margin))
    else:
        value = np.where(in_bounds, 1., 0.)
    return value

class ImitationTask(object):
  """Imitation reference motion task."""

  def __init__(self,
               visualize,
               weight=1.0,
               terminal_condition=imitation_terminal_conditions.imitation_terminal_condition,
               mode="train",
               ref_motion_filenames=None,
               enable_cycle_sync=True,
               enable_sync_root_rotation=False,
               clip_velocity=None,
               tar_frame_steps=None,
               clip_time_min=np.inf,
               clip_time_max=np.inf,
               ref_state_init_prob=1.0,
               enable_rand_init_time=True,
               enable_phase_only=False,
               warmup_time=0.0,
               pose_weight=0.6,
               velocity_weight=0.1,
               end_effector_weight=0.2,
               root_pose_weight=0.15,
               root_velocity_weight=0.15,
               pose_err_scale=5.0,
               velocity_err_scale=0.01, #altered from original paper
               end_effector_err_scale=40,
               end_effector_height_err_scale=3.0,
               root_pose_err_scale=20,
               root_velocity_err_scale=2,
               perturb_init_state_prob=0.0,
               tar_obs_noise=None,
               draw_ref_model_alpha=0.5):
    """Initializes the task.

    Args:
      weight: Float. The scaling factor for the reward.
      terminal_condition: Callable object or function. Determines if the task is
        done.
      ref_motion_filenames: List of files containing reference motion data.
      enable_cycle_sync: Boolean indicating if the root of the reference motion
        should be synchronized with the root of the simulated robot at the start
        of every cycle to mitigate drift.
      clip_velocity: if not None, we will clip the velocity with this value.
      tar_frame_steps: The number of steps to sample each target frame to
        include in the target observations.
      clip_time_min: Minimum amount of time a reference motion clip is active
        before switching to another clip.
      clip_time_max: Maximum amount of time a reference motion clip is active
        before switching to another clip.
      ref_state_init_prob: Probability of initializing the robot to a state from
        the reference at the start of each episode. When not initializing to a
        reference state, the robot will be initialized to the default state.
      enable_rand_init_time: Flag for enabling randomly initializing to
        different points in time alont a reference motion.
      warmup_time: Amount of time for the robot to move from default pose to
        reference pose at the start of each episode. This helps for deployment,
        so that the robot doesn't try to move to the reference pose too quickly
        from its default pose.
      pose_weight: Pose reward weight.
      velocity_weight: Velocity reward weight.
      end_effector_weight: End effector reward weight.
      root_pose_weight: Root position and rotation reward weight.
      root_velocity_weight: Root linear and angular velocity reward weight.
      pose_err_scale: Pose error scale for calculating pose reward.
      velocity_err_scale: Velocity error scale for calculating velocity reward.
      end_effector_err_scale: End effector error scale for calculating end
        effector reward.
      end_effector_height_err_scale: End effector height error scale for
        calculating the end effector reward.
      root_pose_err_scale: Root position and rotation error scale for
        calculating root position and rotation reward.
      root_velocity_err_scale: Root linear and angular velocity error scale for
        calculating root linear and angular velocity reward.
      perturb_init_state_prob: Probability of applying random pertubations to
        the initial state.
      tar_obs_noise: List of the standard deviations of the noise to be applied
        to the target observations [base rotation std, base position std].
      draw_ref_model_alpha: Color transparency for drawing the reference model.
    """
    self._visualize = visualize
    self._weight = weight
    self._terminal_condition = terminal_condition
    self._mode=mode
    self._last_base_position = None
    self._clip_velocity = clip_velocity
    self._action_history_sensor = None
    self._env = None

    assert ref_motion_filenames is not None
    self._ref_state_init_prob = ref_state_init_prob
    self._enable_rand_init_time = enable_rand_init_time
    self._enable_phase_only = enable_phase_only
    self._warmup_time = warmup_time
    self._curr_episode_warmup = False
    self._tar_frame_steps = tar_frame_steps
    self._ref_motion_filenames = ref_motion_filenames
    self._ref_motions = None

    if self._tar_frame_steps is None:
      self._tar_frame_steps = [1, 2]

    self._clip_time_min = clip_time_min
    self._clip_time_max = clip_time_max
    self._clip_change_time = clip_time_min

    self._enable_cycle_sync = enable_cycle_sync
    self._enable_sync_root_rotation=enable_sync_root_rotation
    self._active_motion_id = -1
    self._motion_time_offset = 0.0
    self._episode_start_time_offset = 0.0
    self._ref_model = None
    self._ref_pose = None
    self._ref_vel = None
    self._default_pose = None
    self._perturb_init_state_prob = perturb_init_state_prob
    self._tar_obs_noise = tar_obs_noise
    self._draw_ref_model_alpha = draw_ref_model_alpha

    self._prev_motion_phase = 0
    self._origin_offset_rot = np.array([0, 0, 0, 1])
    self._origin_offset_pos = np.zeros(motion_data.MotionData.POS_SIZE)

    # reward function parameters
    self._pose_weight = pose_weight
    self._velocity_weight = velocity_weight
    self._end_effector_weight = end_effector_weight
    self._root_pose_weight = root_pose_weight
    self._root_velocity_weight = root_velocity_weight

    self._pose_err_scale = pose_err_scale
    self._velocity_err_scale = velocity_err_scale
    self._end_effector_err_scale = end_effector_err_scale
    self._end_effector_height_err_scale = end_effector_height_err_scale
    self._root_pose_err_scale = root_pose_err_scale
    self._root_velocity_err_scale = root_velocity_err_scale

    return

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env

    self._last_base_position = self._get_sim_base_position()
    self._episode_start_time_offset = 0.0

    if (self._ref_motions is None or self._env.hard_reset):
      self._ref_motions = self._load_ref_motions(self._ref_motion_filenames)
      self._active_motion_id = self._sample_ref_motion()

    if (self._ref_model is None or self._env.hard_reset):
      self._build_joint_data()

    if self._default_pose is None or self._env.hard_reset:
      self._default_pose = self._record_default_pose()

    self._curr_episode_warmup = False
    self._reset_ref_motion()
    return

  def update(self, env):
    """Updates the internal state of the task."""
    del env

    self._update_ref_motion()
    self._last_base_position = self._get_sim_base_position()

    return

  def done(self, env):
    """Checks if the episode is over."""
    del env
    done = self._terminal_condition(self._env,self._mode)

    return done

  def get_num_motions(self):
    """Get the number of reference motions to be imitated.

    Returns:
      Number of reference motions.
    """
    return len(self._ref_motions)

  def get_num_tar_frames(self):
    """Get the number of target frames to include in the observations.

    Returns:
      Number of target frames.
    """
    return len(self._tar_frame_steps)

  def is_motion_over(self):
    """Checks if the current reference motion is over.

    Returns:
      Boolean indicating if the motion is over.
    """
    time = self._get_motion_time()
    motion = self.get_active_motion()
    is_over = motion.is_over(time)
    return is_over

  def get_active_motion(self):
    """Get index of the active reference motion currently being imitated.

    Returns:
      Index of the active reference motion.
    """
    return self._ref_motions[self._active_motion_id]

  def build_target_obs(self, use_phase=True):
    """Constructs the target observations, consisting of a sequence of

    target frames for future timesteps. The tartet poses to include is
    specified by self._tar_frame_steps.

    Returns:
      An array containing the target frames.
    """
    return np.array([TARGET_VELOCITY])
  def get_target_obs_bounds(self, use_phase=True):
    """Get bounds for target observations.

    Returns:
      low: Array containing the minimum value for each target observation
        features.
      high: Array containing the maximum value for each target observation
        features.
    """
    if self._enable_phase_only:
      return np.zeros((1)), np.ones((1))

    pos_bound = 2 * np.ones(motion_data.MotionData.POS_SIZE)
    rot_bound = 1 * np.ones(motion_data.MotionData.ROT_SIZE)

    pose_size = self.get_pose_size()
    low = np.inf * np.ones(pose_size)
    high = -np.inf * np.ones(pose_size)

    num_tar_frames = self.get_num_tar_frames()
    low = np.concatenate([low] * num_tar_frames, axis=-1)
    high = np.concatenate([high] * num_tar_frames, axis=-1)

    return low, high

  def set_ref_state_init_prob(self, prob):
    self._ref_state_init_prob = prob
    return

  def reward(self, env):
    """Get the reward without side effects."""
    del env
    loco_reward = self.custom_reward_loco(TARGET_VELOCITY, LOCO_SLOPE)
    energy_penalty = self.custom_energy_penalty(ENERGY_EXP_SCALE)
    pose_reward = self.custom_pose_reward(POSE_SCALING)
    height_reward = self.custom_height_reward(WALKING_MIN_HEIGHT, HEIGHT_SLOPE)
    deviation_penalty = self.custom_deviation_penalty(DEVIATION_SCALING)
    angle_penalty = self.custom_joint_angle_penalty(ANGLE_SCALING)

    rew = np.array([loco_reward, energy_penalty, pose_reward, height_reward, deviation_penalty, angle_penalty])
    reward = np.dot(rew, SUBREWARD_WEIGHTS)

    if self._visualize and 1:
      print(f"[{loco_reward:3.2f} {energy_penalty:3.2f} {pose_reward:3.2f} {height_reward:3.2f} {deviation_penalty:3.2f} {angle_penalty:3.2f}]--> {reward:3.2f}")

    self._env.aggregate_returns = rew
    return reward * self._weight
  
  def custom_reward_loco(self, tar_speed, slope):
    """Return the locomotion reward as according to the walk-in-the park
    
    This function is normalized at [0, 1]
    """
    env = self._env
    robot = env.robot
    sim_model = robot.quadruped
    pyb = self._get_pybullet_client()

    root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)
    root_vel_sim = np.array(root_vel_sim)
    tar_dir_speed = root_vel_sim[0]

    rewards = my_tolerance(tar_dir_speed, tar_speed, VEL_UPPER_BOUND, slope*tar_speed, 0.)
    return rewards

  def custom_energy_penalty(self, scaling):
    """Returns the energy penalty in terms of motor angular velocities and torques
    
    Need to check the normalization of this func
    """
    env = self._env
    robot = env.robot
    energy_consumption = - np.sum(np.abs(robot.GetMotorTorques() * robot.GetMotorVelocities()))
    return np.exp(scaling * energy_consumption)

  def custom_pose_reward(self, scaling):
    """Rewards an upright cheetah. Ignored roll since it is needed for movement?
    
    """
    env = self._env
    robot = env.robot
    roll_pitch_yaw = robot.GetTrueBaseRollPitchYaw()
    roll = roll_pitch_yaw[0]
    pitch = roll_pitch_yaw[1]
    yaw = roll_pitch_yaw[2]
    reward = np.abs(yaw) + np.abs(pitch) + 0.5*np.abs(roll)
    return np.exp(- scaling * reward)
  
  def custom_height_reward(self, min_height, slope):
    env = self._env
    robot = env.robot
    sim_model = robot.quadruped
    pyb = env._pybullet_client
    root_pos_sim, root_rot_sim = pyb.getBasePositionAndOrientation(
    env.robot.quadruped)
    return my_tolerance(root_pos_sim[2], min_height, 2*min_height, slope*min_height, 0)
  
  def custom_deviation_penalty(self, scaling):
    env = self._env
    robot = env.robot
    sim_model = robot.quadruped
    pyb = env._pybullet_client
    root_vel_sim, root_ang_vel_sim = pyb.getBaseVelocity(sim_model)
    return np.exp(- scaling * root_vel_sim[1]*root_vel_sim[1])

  def custom_joint_angle_penalty(self, scaling):
    env = self._env
    robot = env.robot
    motor_angles = np.array([state[0] for state in robot._joint_states])
    init_angles= robot.GetDefaultInitJointPose()
    delta = np.sum(np.abs(motor_angles - init_angles))
    return np.exp(-scaling * delta)


  def _load_ref_motions(self, filenames):
    """Load reference motions.

    Args:
      dir: Directory containing the reference motion files.
      filenames: Names of files in dir to be loaded.
    Returns: List of reference motions loaded from the files.
    """
    num_files = len(filenames)
    if num_files == 0:
      raise ValueError("No reference motions specified.")

    total_time = 0.0
    motions = []
    for filename in filenames:
      curr_motion = motion_data.MotionData(filename)

      curr_duration = curr_motion.get_duration()
      total_time += curr_duration
      motions.append(curr_motion)

    logging.info("Loaded {:d} motion clips with {:.3f}s of motion data.".format(
        num_files, total_time))

    return motions

  def _build_ref_model(self):
    """Constructs simulated model for playing back the reference motion.

    Returns:
      Handle to the simulated model for the reference motion.
    """
    ref_col = [1, 1, 1, self._draw_ref_model_alpha]

    pyb = self._get_pybullet_client()
    urdf_file = self._env.robot.GetURDFFile()
    ref_model = pyb.loadURDF(urdf_file, useFixedBase=True)

    pyb.changeDynamics(ref_model, -1, linearDamping=0, angularDamping=0)

    pyb.setCollisionFilterGroupMask(
        ref_model, -1, collisionFilterGroup=0, collisionFilterMask=0)

    pyb.changeDynamics(
        ref_model,
        -1,
        activationState=pyb.ACTIVATION_STATE_SLEEP +
        pyb.ACTIVATION_STATE_ENABLE_SLEEPING +
        pyb.ACTIVATION_STATE_DISABLE_WAKEUP)

    pyb.changeVisualShape(ref_model, -1, rgbaColor=ref_col)

    num_joints = pyb.getNumJoints(ref_model)
    num_joints_sim = pyb.getNumJoints(self._env.robot.quadruped)
    assert (
        num_joints == num_joints_sim
    ), "ref model must have the same number of joints as the simulated model."

    for j in range(num_joints):
      pyb.setCollisionFilterGroupMask(
          ref_model, j, collisionFilterGroup=0, collisionFilterMask=0)

      pyb.changeDynamics(
          ref_model,
          j,
          activationState=pyb.ACTIVATION_STATE_SLEEP +
          pyb.ACTIVATION_STATE_ENABLE_SLEEPING +
          pyb.ACTIVATION_STATE_DISABLE_WAKEUP)

      pyb.changeVisualShape(ref_model, j, rgbaColor=ref_col)

    return ref_model

  def _build_joint_data(self):
    """Precomputes joint data to facilitating accessing data from motion frames."""
    num_joints = self._get_num_joints()
    self._joint_pose_idx = np.zeros(num_joints, dtype=np.int32)
    self._joint_pose_size = np.zeros(num_joints, dtype=np.int32)
    self._joint_vel_idx = np.zeros(num_joints, dtype=np.int32)
    self._joint_vel_size = np.zeros(num_joints, dtype=np.int32)

    for j in range(num_joints):
      pyb = self._get_pybullet_client()
      j_info = pyb.getJointInfo(self._env.robot.quadruped, j)
      j_state = pyb.getJointStateMultiDof(self._env.robot.quadruped, j)

      j_pose_idx = j_info[3]
      j_vel_idx = j_info[4]
      j_pose_size = len(j_state[0])
      j_vel_size = len(j_state[1])

      if (j_pose_idx < 0):
        assert (j_vel_idx < 0)
        assert (j_pose_size == 0)
        assert (j_vel_size == 0)

        if (j == 0):
          j_pose_idx = 0
          j_vel_idx = 0
        else:
          j_pose_idx = self._joint_pose_idx[j - 1] + self._joint_pose_size[j -
                                                                           1]
          j_vel_idx = self._joint_vel_idx[j - 1] + self._joint_vel_size[j - 1]

      self._joint_pose_idx[j] = j_pose_idx
      self._joint_pose_size[j] = j_pose_size
      self._joint_vel_idx[j] = j_vel_idx
      self._joint_vel_size[j] = j_vel_size

    return

  def _reset_ref_motion(self):
    """Reset reference motion.

    First randomly select a new reference motion from
    the set of available motions, and then resets to a random point along the
    selected motion.
    """
    self._active_motion_id = self._sample_ref_motion()
    self._origin_offset_rot = np.array([0, 0, 0, 1])
    self._origin_offset_pos.fill(0.0)

    self._reset_motion_time_offset()
    motion = self.get_active_motion()
    time = self._get_motion_time()

    ref_pose = self._calc_ref_pose(time)
    ref_root_pos = motion.get_frame_root_pos(ref_pose)
    ref_root_rot = motion.get_frame_root_rot(ref_pose)
    sim_root_pos = self._get_sim_base_position()
    sim_root_rot = self._get_sim_base_rotation()

    # move the root to the same position and rotation as simulated robot
    self._origin_offset_pos = sim_root_pos - ref_root_pos
    self._origin_offset_pos[2] = 0

    ref_heading = motion_util.calc_heading(ref_root_rot)
    sim_heading = motion_util.calc_heading(sim_root_rot)
    delta_heading = sim_heading - ref_heading
    self._origin_offset_rot = transformations.quaternion_about_axis(
        delta_heading, [0, 0, 1])

    self._ref_pose = self._calc_ref_pose(time)
    self._ref_vel = self._calc_ref_vel(time)
    self._update_ref_model()

    self._prev_motion_phase = motion.calc_phase(time)
    self._reset_clip_change_time()

    return

  def _update_ref_motion(self):
    """Updates the reference motion and synchronizes the state of the reference

    model with the current motion frame.
    """


    return


  def _update_ref_model(self):
    """Synchronizes the reference model to the pose and velocity of the

    reference motion. Only done if visualization is enabled.
    """
    return

  def _sync_sim_model(self, perturb_state):
    """Synchronizes the simulated character to the pose and velocity of the

    reference motion.

    Args:
      perturb_state: A flag for enabling perturbations to be applied to state.
    """
    pose = self._ref_pose
    vel = self._ref_vel

    self._set_state(self._env.robot.quadruped, pose, vel)
    self._env.robot.ReceiveObservation()
    return

  def _set_state(self, phys_model, pose, vel):
    """Set the state of a character to the given pose and velocity.

    Args:
      phys_model: handle of the character
      pose: pose to be applied to the character
      vel: velocity to be applied to the character
    """
    motion = self.get_active_motion()
    pyb = self._get_pybullet_client()

    root_pos = motion.get_frame_root_pos(pose)
    root_rot = motion.get_frame_root_rot(pose)
    root_vel = motion.get_frame_root_vel(vel)
    root_ang_vel = motion.get_frame_root_ang_vel(vel)

    pyb.resetBasePositionAndOrientation(phys_model, root_pos, root_rot)
    pyb.resetBaseVelocity(phys_model, root_vel, root_ang_vel)

    num_joints = self._get_num_joints()
    for j in range(num_joints):
      q_idx = self._get_joint_pose_idx(j)
      q_size = self._get_joint_pose_size(j)

      dq_idx = self._get_joint_vel_idx(j)
      dq_size = self._get_joint_vel_size(j)

      if (q_size > 0):
        assert (dq_size > 0)

        j_pose = pose[q_idx:(q_idx + q_size)]
        j_vel = vel[dq_idx:(dq_idx + dq_size)]
        pyb.resetJointStateMultiDof(phys_model, j, j_pose, j_vel)

    return

  def _get_pybullet_client(self):
    """Get bullet client from the environment"""
    return self._env._pybullet_client

  def _get_motion_time(self):
    """Get the time since the start of the reference motion."""
    time = self._env.get_time_since_reset()

    # Needed to ensure that during deployment, the first timestep will be at
    # time = 0
    if self._env.env_step_counter == 0:
      self._episode_start_time_offset = -time

    time += self._motion_time_offset
    time += self._episode_start_time_offset

    if self._curr_episode_warmup:
      # if warmup is enabled, then apply a time offset to give the robot more
      # time to move to the reference motion
      time -= self._warmup_time

    return time

  def _get_num_joints(self):
    """Get the number of joints in the character's body."""
    pyb = self._get_pybullet_client()
    return pyb.getNumJoints(self._env.robot.quadruped)

  def _get_joint_pose_idx(self, j):
    """Get the starting index of the pose data for a give joint in a pose array."""
    idx = self._joint_pose_idx[j]
    return idx

  def _get_joint_vel_idx(self, j):
    """Get the starting index of the velocity data for a give joint in a

    velocity array.
    """
    idx = self._joint_vel_idx[j]
    return idx

  def _get_joint_pose_size(self, j):
    """Get the size of the pose data for a give joint in a pose array."""
    pose_size = self._joint_pose_size[j]
    assert (pose_size == 1 or
            pose_size == 0), "Only support 1D and 0D joints at the moment."
    return pose_size

  def _get_joint_vel_size(self, j):
    """Get the size of the velocity data for a give joint in a velocity array."""
    vel_size = self._joint_vel_size[j]
    assert (vel_size == 1 or
            vel_size == 0), "Only support 1D and 0D joints at the moment."
    return vel_size

  def get_pose_size(self):
    """Get the total size of a pose array."""
    num_joints = self._get_num_joints()
    pose_size = self._get_joint_pose_idx(
        num_joints - 1) + self._get_joint_pose_size(num_joints - 1)
    return pose_size

  def get_vel_size(self):
    """Get the total size of a velocity array."""
    num_joints = self._get_num_joints()
    vel_size = self._get_joint_vel_idx(
        num_joints - 1) + self._get_joint_vel_size(num_joints - 1)
    return vel_size

  def _get_sim_base_position(self):
    pyb = self._get_pybullet_client()
    pos = pyb.getBasePositionAndOrientation(self._env.robot.quadruped)[0]
    pos = np.array(pos)
    return pos

  def _get_sim_base_rotation(self):
    pyb = self._get_pybullet_client()
    rotation = pyb.getBasePositionAndOrientation(self._env.robot.quadruped)[1]
    rotation = np.array(rotation)
    return rotation

  def _get_ref_base_position(self):
    return self._ref_pose[:3]

  def _get_ref_base_rotation(self):
    return self._ref_pose[3:7]

  def get_ref_base_position(self):
    return self._get_ref_base_position()

  def get_ref_base_rotation(self):
    return self._get_ref_base_rotation()

  def _calc_ref_pose(self, time, apply_origin_offset=True):
    """Calculates the reference pose for a given point in time.

    Args:
      time: Time elapsed since the start of the reference motion.
      apply_origin_offset: A flag for enabling the origin offset to be applied
        to the pose.

    Returns:
      An array containing the reference pose at the given point in time.
    """

    motion = self.get_active_motion()
    enable_warmup_pose = self._curr_episode_warmup \
                         and time >= -self._warmup_time and time < 0.0
    if enable_warmup_pose:
      pose = self._calc_ref_pose_warmup()
    else:
      pose = motion.calc_frame(time)

    if apply_origin_offset:
      root_pos = motion.get_frame_root_pos(pose)
      root_rot = motion.get_frame_root_rot(pose)

      root_rot = transformations.quaternion_multiply(self._origin_offset_rot,
                                                     root_rot)
      root_pos = pose3d.QuaternionRotatePoint(root_pos, self._origin_offset_rot)
      root_pos += self._origin_offset_pos

      motion.set_frame_root_rot(root_rot, pose)
      motion.set_frame_root_pos(root_pos, pose)

    return pose

  def _calc_ref_vel(self, time):
    """Calculates the reference velocity for a given point in time.

    Args:
      time: Time elapsed since the start of the reference motion.

    Returns:
      An array containing the reference velocity at the given point in time.
    """
    motion = self.get_active_motion()
    enable_warmup_pose = self._curr_episode_warmup \
                         and time >= -self._warmup_time and time < 0.0
    if enable_warmup_pose:
      vel = self._calc_ref_vel_warmup()
    else:
      vel = motion.calc_frame_vel(time)

    root_vel = motion.get_frame_root_vel(vel)
    root_ang_vel = motion.get_frame_root_ang_vel(vel)

    root_vel = pose3d.QuaternionRotatePoint(root_vel, self._origin_offset_rot)
    root_ang_vel = pose3d.QuaternionRotatePoint(root_ang_vel,
                                                self._origin_offset_rot)

    motion.set_frame_root_vel(root_vel, vel)
    motion.set_frame_root_ang_vel(root_ang_vel, vel)

    return vel

  def _calc_ref_pose_warmup(self):
    """Calculate default reference  pose during warmup period."""
    motion = self.get_active_motion()
    pose0 = motion.calc_frame(0)
    warmup_pose = self._default_pose.copy()

    pose_root_rot = motion.get_frame_root_rot(pose0)
    default_root_rot = motion.get_frame_root_rot(warmup_pose)
    default_root_pos = motion.get_frame_root_pos(warmup_pose)

    pose_heading = motion_util.calc_heading(pose_root_rot)
    default_heading = motion_util.calc_heading(default_root_rot)
    delta_heading = pose_heading - default_heading
    delta_heading_rot = transformations.quaternion_about_axis(
        delta_heading, [0, 0, 1])

    default_root_pos = pose3d.QuaternionRotatePoint(default_root_pos,
                                                    delta_heading_rot)
    default_root_rot = transformations.quaternion_multiply(
        delta_heading_rot, default_root_rot)

    motion.set_frame_root_pos(default_root_pos, warmup_pose)
    motion.set_frame_root_rot(default_root_rot, warmup_pose)

    return warmup_pose

  def _calc_ref_vel_warmup(self):
    """Calculate default reference velocity during warmup period."""

    # set target velocity to zero to encourage smoother motion during
    # transition to the reference motion.
    vel_size = self.get_vel_size()
    vel = np.zeros(vel_size)
    return vel

  def _sync_ref_origin(self, sync_root_position, sync_root_rotation):
    """Moves the origin of the reference motion, such that the root of the

    simulated and reference characters are at the same location. This is used
    to periodically synchronize the reference motion with the simulated
    character in order to mitigate drift.

    Args:
      sync_root_position: boolean indicating if the root position should be
        synchronized
      sync_root_rotation: boolean indicating if the root rotation should be
        synchronized
    """
    time = self._get_motion_time()
    motion = self.get_active_motion()
    ref_pose = self._calc_ref_pose(time, apply_origin_offset=False)

    if sync_root_rotation:
      ref_rot = motion.get_frame_root_rot(ref_pose)
      sim_rot = self._get_sim_base_rotation()
      ref_heading = self._calc_heading(ref_rot)
      sim_heading = self._calc_heading(sim_rot)
      heading_diff = sim_heading - ref_heading

      self._origin_offset_rot = transformations.quaternion_about_axis(
          heading_diff, [0, 0, 1])

    if sync_root_position:
      ref_pos = motion.get_frame_root_pos(ref_pose)
      ref_pos = pose3d.QuaternionRotatePoint(ref_pos, self._origin_offset_rot)
      sim_pos = self._get_sim_base_position()
      self._origin_offset_pos = sim_pos - ref_pos
      self._origin_offset_pos[2] = 0  # only sync along horizontal plane

    return

  def _reset_clip_change_time(self):
    """Reset the time when the current motion clip is changed to a new one."""
    if np.isfinite(self._clip_time_min) and np.isfinite(self._clip_time_max):
      clip_dur = self._rand_uniform(self._clip_time_min, self._clip_time_max)
      time = self._get_motion_time()
      change_time = time + clip_dur
    else:
      change_time = np.inf

    self._clip_change_time = change_time

    return

  def _build_sim_pose(self, phys_model):
    """Build  pose vector from simulated model."""
    pose = np.zeros(self.get_pose_size())
    pyb = self._get_pybullet_client()
    root_pos, root_rot = pyb.getBasePositionAndOrientation(phys_model)
    root_pos = np.array(root_pos)
    root_rot = np.array(root_rot)

    joint_pose = []

    num_joints = self._get_num_joints()
    for j in range(num_joints):
      j_state_sim = pyb.getJointStateMultiDof(phys_model, j)
      j_pose_sim = np.array(j_state_sim[0])

      j_size_sim = len(j_pose_sim)

      if j_size_sim > 0:
        joint_pose.append(j_pose_sim)

    joint_pose = np.concatenate(joint_pose)

    motion = self.get_active_motion()
    motion.set_frame_root_pos(root_pos, pose)
    motion.set_frame_root_rot(root_rot, pose)
    motion.set_frame_joints(joint_pose, pose)

    return pose

  def _get_default_root_rotation(self):
    """Get default root rotation."""
    motion = self.get_active_motion()
    root_rot = motion.get_frame_root_rot(self._default_pose)
    return root_rot

  def _sample_ref_motion(self):
    """Samples a motion ID randomly from the set of reference motions.

    Returns:
      ID of the selected motion.
    """
    num_motions = self.get_num_motions()
    motion_id = self._randint(0, num_motions)
    return motion_id

  def _change_ref_motion(self, motion_id):
    """Change the current active reference motion to a specified motion.

    Args:
      motion_id: ID of new motion.
    """
    self._active_motion_id = motion_id
    return

  def _check_change_clip(self):
    """Check if the current motion clip should be changed."""
    time = self._get_motion_time()
    num_motions = self.get_num_motions()
    change = (time >= self._clip_change_time) and (num_motions > 1)
    return change

  def _reset_motion_time_offset(self):
    self._motion_time_offset = 0.0
    return

  def _sample_time_offset(self):
    """Sample a random time offset for the currently active motion.

    Returns:
      A random time offset between 0 and the duration of the currently active
      motion (in seconds).
    """
    motion = self.get_active_motion()
    dur = motion.get_duration()
    offset = self._rand_uniform(0, dur)

    return offset

  def _rand_uniform(self, val_min, val_max, size=None):
    """Samples random float between [val_min, val_max]."""
    if hasattr(self._env, "np_random"):
      rand_val = self._env.np_random.uniform(val_min, val_max, size=size)
    else:
      rand_val = np.random.uniform(val_min, val_max, size=size)
    return rand_val

  def _randint(self, val_min, val_max, size=None):
    """Samples random integer between [val_min, val_max]."""
    if hasattr(self._env, "np_random"):
      rand_val = self._env.np_random.randint(val_min, val_max, size=size)
    else:
      rand_val = np.random.randint(val_min, val_max, size=size)
    return rand_val

  def _randn(self, mean, std, size=None):
    """Samples random value from gaussian."""

    if size is None:
      size = []

    if hasattr(self._env, "np_random"):
      rand_val = self._env.np_random.randn(*size)
    else:
      rand_val = np.random.randn(*size)

    rand_val = std * rand_val + mean
    return rand_val

  def _calc_heading(self, root_rotation):
    """Returns the heading of a rotation q, specified as a quaternion.

    The heading represents the rotational component of q along the vertical
    axis (z axis). The heading is computing with respect to the robot's default
    root orientation (self._default_root_rotation). This is because different
    models have different coordinate systems, and some models may not have the z
    axis as the up direction. This is similar to robot.GetTrueBaseOrientation(),
    but is applied both to the robot and reference motion.

    Args:
      root_rotation: A quaternion representing the rotation of the robot's root.

    Returns:
      An angle representing the rotation about the z axis with respect to
      the default orientation.

    """
    inv_default_rotation = transformations.quaternion_conjugate(
        self._get_default_root_rotation())
    rel_rotation = transformations.quaternion_multiply(root_rotation,
                                                       inv_default_rotation)
    heading = motion_util.calc_heading(rel_rotation)
    return heading

  def _calc_heading_rot(self, root_rotation):
    """Return a quaternion representing the heading rotation of q along the vertical axis (z axis).

    The heading is computing with respect to the robot's default root
    orientation (self._default_root_rotation). This is because different models
    have different coordinate systems, and some models may not have the z axis
    as the up direction.

    Args:
      root_rotation: A quaternion representing the rotation of the robot's root.

    Returns:
      A quaternion representing the rotation about the z axis with respect to
      the default orientation.

    """
    inv_default_rotation = transformations.quaternion_conjugate(
        self._get_default_root_rotation())
    rel_rotation = transformations.quaternion_multiply(root_rotation,
                                                       inv_default_rotation)
    heading_rot = motion_util.calc_heading_rot(rel_rotation)
    return heading_rot

  def _enable_warmup(self):
    """Check if warmup period is enabled at start of each episode."""
    return self._warmup_time > 0

  def _enable_perturb_init_state(self):
    """Check if initial state perturbations are enabled."""
    return self._perturb_init_state_prob > 0.0

  def _apply_state_perturb(self, pose, vel):
    """Apply random perturbations to the state pose and velocities."""
    root_pos_std = 0.025
    root_rot_std = 0.025 * np.pi
    joint_pose_std = 0.05 * np.pi
    root_vel_std = 0.1
    root_ang_vel_std = 0.05 * np.pi
    joint_vel_std = 0.05 * np.pi

    perturb_pose = pose.copy()
    perturb_vel = vel.copy()

    motion = self.get_active_motion()
    root_pos = motion.get_frame_root_pos(perturb_pose)
    root_rot = motion.get_frame_root_rot(perturb_pose)
    joint_pose = motion.get_frame_joints(perturb_pose)
    root_vel = motion.get_frame_root_vel(perturb_vel)
    root_ang_vel = motion.get_frame_root_ang_vel(perturb_vel)
    joint_vel = motion.get_frame_joints_vel(perturb_vel)

    root_pos[0] += self._randn(0, root_pos_std)
    root_pos[1] += self._randn(0, root_pos_std)

    rand_axis = self._rand_uniform(-1, 1, size=[3])
    rand_axis /= np.linalg.norm(rand_axis)
    rand_theta = self._randn(0, root_rot_std)
    rand_rot = transformations.quaternion_about_axis(rand_theta, rand_axis)
    root_rot = transformations.quaternion_multiply(rand_rot, root_rot)

    joint_pose += self._randn(0, joint_pose_std, size=joint_pose.shape)

    root_vel[0] += self._randn(0, root_vel_std)
    root_vel[1] += self._randn(0, root_vel_std)
    root_ang_vel += self._randn(0, root_ang_vel_std, size=root_ang_vel.shape)
    joint_vel += self._randn(0, joint_vel_std, size=joint_vel.shape)

    motion.set_frame_root_pos(root_pos, perturb_pose)
    motion.set_frame_root_rot(root_rot, perturb_pose)
    motion.set_frame_joints(joint_pose, perturb_pose)
    motion.set_frame_root_vel(root_vel, perturb_vel)
    motion.set_frame_root_ang_vel(root_ang_vel, perturb_vel)
    motion.set_frame_joints_vel(joint_vel, perturb_vel)

    return perturb_pose, perturb_vel

  def _record_default_pose(self):
    root_pos = self._env.robot.GetDefaultInitPosition()
    root_rot = self._env.robot.GetDefaultInitOrientation()
    joint_pose = self._env.robot.GetDefaultInitJointPose()

    pose = np.concatenate([root_pos, root_rot, joint_pose])

    return pose

  def _robot_scale(self):
    return 1.0
