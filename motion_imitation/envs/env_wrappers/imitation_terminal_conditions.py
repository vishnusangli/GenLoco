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

"""Contains the terminal conditions for imitation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from utilities import pose3d
from utilities import motion_util
from pybullet_utils import transformations

TARGET_VELOCITY = 0.6
TIME_DELAY=1.0

"""
Bgeinning Training:
Target Velocity: 0.6
Tolerance:

Walking Training

"""
def imitation_terminal_condition(env,
                                 mode,
                                 dist_fail_threshold=0.3,
                                 rot_fail_threshold= np.pi*1/10):
  """A terminal condition for motion imitation task.

  Args:
    env: An instance of MinitaurGymEnv
    dist_fail_threshold: Max distance the simulated character's root is allowed
      to drift from the reference motion before the episode terminates.
    rot_fail_threshold: Max rotational difference between simulated character's
      root and the reference motion's root before the episode terminates.

  Returns:
    A boolean indicating if episode is over.
  """

  pyb = env._pybullet_client
  task = env._task

  motion_over = task.is_motion_over()
  foot_links = env.robot.GetFootLinkIDs()
  ground = env.get_ground()

  contact_fall = False
  # sometimes the robot can be initialized with some ground penetration
  # so do not check for contacts until after the first env step.
  if env.env_step_counter > 0:
    robot_ground_contacts = env.pybullet_client.getContactPoints(
        bodyA=env.robot.quadruped, bodyB=ground)

    for contact in robot_ground_contacts:
      if contact[3] not in foot_links:
        contact_fall = True
        break

  root_pos_ref, root_rot_ref = tuple(task.get_ref_base_position()), tuple(task.get_ref_base_rotation())
  env_time = task._get_motion_time()
  root_pos_ref = np.array([TARGET_VELOCITY * (env_time-TIME_DELAY), 0, 0])

  root_pos_sim, root_rot_sim = pyb.getBasePositionAndOrientation(
      env.robot.quadruped)

  #Position Error Termination
  # X-direction (array index 0) is the desired moving direction
  # Check whether x distance is greater or within dist_fail_threshold of TARGET_VELOCITY-covered distance
  # in similar time.

  # Then check squared distance in other directions and check whether they are under dist_fail_threshold
  root_pos_diff = np.array(root_pos_ref) - np.array(root_pos_sim)
  xdir_pos_fail = root_pos_diff[0] > -dist_fail_threshold
  other_pos_fail = (root_pos_diff[1]*root_pos_diff[1])+(root_pos_diff[2]*root_pos_diff[2]) > 0.09
  root_pos_fail = xdir_pos_fail or other_pos_fail
  if (env_time<=TIME_DELAY): root_pos_fail=False

  roll_pitch_yaw = np.array(env.robot.GetTrueBaseRollPitchYaw())
  root_rot_fail = np.any(roll_pitch_yaw > rot_fail_threshold)

  # root_rot_diff = transformations.quaternion_multiply(
  #     np.array(root_rot_ref),
  #     transformations.quaternion_conjugate(np.array(root_rot_sim)))
  # _, root_rot_diff_angle = pose3d.QuaternionToAxisAngle(
  #     root_rot_diff)
  # root_rot_diff_angle = motion_util.normalize_rotation_angle(
  #     root_rot_diff_angle)
  # root_rot_fail = (np.abs(root_rot_diff_angle) > rot_fail_threshold)

  done = root_pos_fail \
      or root_rot_fail \
      or contact_fall
  # if done:
  #   print(f"Fail: {root_rot_fail:1.0f} ({xdir_pos_fail:1.0f}{other_pos_fail:1.0f}) {contact_fall:1.0f} | {task._get_motion_time():8.2f} | {np.array2string(root_pos_diff, precision=1)}")
  # if mode =="test":
  #   done=contact_fall or motion_over
  return done
