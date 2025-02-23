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

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


"""Pybullet simulation of a Laikago robot."""
import math
import re
import numpy as np
import copy
import pybullet as pyb  # pytype: disable=import-error
import collections
from motion_imitation.robots import laikago_pose_utils
from motion_imitation.robots import laikago_constants
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config
from motion_imitation.robots import kinematics
from motion_imitation.robots import action_filter
from motion_imitation import unify
NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.48]
JOINT_DIRECTIONS = np.array([-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1])
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = -0.6
KNEE_JOINT_OFFSET = 0.66
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
TWO_PI = 2 * math.pi
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2]
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2]
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 0.3
HIP_P_GAIN = 220.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 2.0


INIT_MOTOR_ANGLES = np.array([
    laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

_CHASSIS_NAME_PATTERN = re.compile(r"\w+_chassis_\w+")
_MOTOR_NAME_PATTERN = re.compile(r"\w+_hip_motor_\w+")
_KNEE_NAME_PATTERN = re.compile(r"\w+_lower_leg_\w+")
_TOE_NAME_PATTERN = re.compile(r"jtoe\d*")

URDF_FILENAME = "laikago/laikago_toes_limits.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203
SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)
KP = [
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
    HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
    ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
]
KD = [
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
    HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
    ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
]
def MapToMinusPiToPi(angles):
  """Maps a list of angles to [-pi, pi].

  Args:
    angles: A list of angles in rad.

  Returns:
    A list of angle mapped to [-pi, pi].
  """
  mapped_angles = copy.deepcopy(angles)
  for i in range(len(angles)):
    mapped_angles[i] = math.fmod(angles[i], TWO_PI)
    if mapped_angles[i] >= math.pi:
      mapped_angles[i] -= TWO_PI
    elif mapped_angles[i] < -math.pi:
      mapped_angles[i] += TWO_PI
  return mapped_angles

class Base_robot(object):
  """A simulation for the Base_robot robot."""
  NUM_LEGS=NUM_LEGS
  NUM_MOTORS = NUM_MOTORS
  MPC_BODY_MASS = 215/9.8
  MPC_BODY_INERTIA = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447)
  MPC_BODY_HEIGHT = 0.42
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="motor_angle_0",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_1",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_2",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_3",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_4",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_5",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_6",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_7",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_8",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_9",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_10",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_11",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND)
  ]

  def __init__(self,
               pybullet_client,
               num_motors=NUM_MOTORS,
               dofs_per_leg=DOFS_PER_LEG,
               time_step=0.001,
               action_repeat=1,
               self_collision_enabled=False,
               motor_control_mode=robot_config.MotorControlMode.POSITION,
               motor_model_class=laikago_motor.LaikagoMotorModel,
               motor_kp=None,
               motor_kd=None,
               motor_torque_limits=None,
               pd_latency=0.0,
               control_latency=0.0,
               observation_noise_stdev=SENSOR_NOISE_STDDEV,
               motor_overheat_protection=False,
               motor_direction=JOINT_DIRECTIONS,
               motor_offset=JOINT_OFFSETS,
               on_rack=False,
               reset_at_current_position=False,
               sensors=None,
               enable_action_interpolation=False,
               enable_action_filter=False,
               reset_time=-1,
               allow_knee_contact=False,
               enable_clip_motor_commands=True):

    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands
    if motor_kp is None:
        motor_kp = [
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
        HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    if motor_kd is None:
        motor_kd = [
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
        HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]

    self.num_motors = num_motors
    self.num_legs = self.num_motors // dofs_per_leg
    self._pybullet_client = pybullet_client
    self._action_repeat = action_repeat
    self._self_collision_enabled = self_collision_enabled
    self._motor_direction = motor_direction
    self._motor_offset = motor_offset
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._max_force = 3.5
    self._pd_latency = pd_latency
    self._control_latency = control_latency
    self._observation_noise_stdev = observation_noise_stdev
    self._observation_history = collections.deque(maxlen=100)
    self._control_observation = []
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._foot_link_ids = []

    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._reset_at_current_position = reset_at_current_position
    self.SetAllSensors(sensors if sensors is not None else list())
    self._is_safe = True

    self._enable_action_interpolation = enable_action_interpolation
    self._enable_action_filter = enable_action_filter
    self._last_action = None
    self._episode_history = unify.HistoryStepsAPI(self, self._GetMotorNames(), motor_kp, motor_kd)

    if not motor_model_class:
      raise ValueError("Must provide a motor model class!")

    if self._on_rack and self._reset_at_current_position:
      raise ValueError("on_rack and reset_at_current_position "
                       "cannot be enabled together")

    if isinstance(motor_kp, (collections.Sequence, np.ndarray)):
      self._motor_kps = np.asarray(motor_kp)
    else:
      self._motor_kps = np.full(num_motors, motor_kp)

    if isinstance(motor_kd, (collections.Sequence, np.ndarray)):
      self._motor_kds = np.asarray(motor_kd)
    else:
      self._motor_kds = np.full(num_motors, motor_kd)

    if isinstance(motor_torque_limits, (collections.Sequence, np.ndarray)):
      self._motor_torque_limits = np.asarray(motor_torque_limits)
    elif motor_torque_limits is None:
      self._motor_torque_limits = None
    else:
      self._motor_torque_limits = motor_torque_limits

    self._motor_control_mode = motor_control_mode
    self._motor_model = motor_model_class(
        kp=motor_kp,
        kd=motor_kd,
        torque_limits=self._motor_torque_limits,
        num_motors=self.num_motors,
        motor_control_mode=motor_control_mode)

    self.time_step = time_step
    self._step_counter = 0

    # This also includes the time spent during the Reset motion.
    self._state_action_counter = 0
    _, self._init_orientation_inv = self._pybullet_client.invertTransform(
        position=[0, 0, 0], orientation=self._GetDefaultInitOrientation())

    if self._enable_action_filter:
      self._action_filter = self._BuildActionFilter()

    # reset_time=-1.0 means skipping the reset motion.
    # See Reset for more details.
    self.Reset(reset_time=reset_time)
    self.ReceiveObservation()

  def GetTimeSinceReset(self):
    return self._step_counter * self.time_step
  
  def ApplyRandomDOF(self):
    posObj, _ = self._pybullet_client.getBasePositionAndOrientation(self.quadruped)
    randomForces, randomTorques = np.random.uniform(0.0, 10.0, 3), np.random.uniform(0.0, 10.0, 3)
    self._pybullet_client.applyExternalForce(objectUniqueId=self.quadruped, linkIndex=-1, forceObj=randomForces, posObj=posObj, flags=self._pybullet_client.WORLD_FRAME)
    self._pybullet_client.applyExternalTorque(objectUniqueId=self.quadruped, linkIndex=-1, torqueObj=randomTorques, flags=self._pybullet_client.WORLD_FRAME)

  def _StepInternal(self, action, motor_control_mode):
    self.ApplyAction(action, motor_control_mode)
    self._pybullet_client.stepSimulation()
    self.ReceiveObservation()
    self._state_action_counter += 1

  def Step(self, action, control_mode=None):
    """Steps simulation."""
    if self._enable_action_filter:
      action = self._FilterAction(action)
    if control_mode==None:
      control_mode = self._motor_control_mode
    for i in range(self._action_repeat):
      proc_action = self.ProcessAction(action, i)
      self._StepInternal(proc_action, control_mode)
      self._step_counter += 1

    self._last_action = action
    self._episode_history.take_reading(self.GetTrueMotorAngles(), self.GetTrueMotorTorques(), action)

  def Terminate(self):
    pass

  def GetFootLinkIDs(self):
    """Get list of IDs for all foot links."""
    return self._foot_link_ids

  def _RecordMassInfoFromURDF(self):
    """Records the mass information from the URDF file."""
    self._base_mass_urdf = []
    for chassis_id in self._chassis_link_ids:
      self._base_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
    self._leg_masses_urdf = []
    for leg_id in self._leg_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
    for motor_id in self._motor_link_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, motor_id)[0])

  def _RecordInertiaInfoFromURDF(self):
    """Record the inertia of each body from URDF file."""
    self._link_urdf = []
    num_bodies = self._pybullet_client.getNumJoints(self.quadruped)
    for body_id in range(-1, num_bodies):  # -1 is for the base link.
      inertia = self._pybullet_client.getDynamicsInfo(self.quadruped,
                                                      body_id)[2]
      self._link_urdf.append(inertia)
    # We need to use id+1 to index self._link_urdf because it has the base
    # (index = -1) at the first element.
    self._base_inertia_urdf = [
        self._link_urdf[chassis_id + 1]
        for chassis_id in self._chassis_link_ids
    ]
    self._leg_inertia_urdf = [
        self._link_urdf[leg_id + 1] for leg_id in self._leg_link_ids
    ]
    self._leg_inertia_urdf.extend(
        [self._link_urdf[motor_id + 1] for motor_id in self._motor_link_ids])

  def _BuildJointNameToIdDict(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _RemoveDefaultJointDamping(self):
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._pybullet_client.changeDynamics(joint_info[0],
                                           -1,
                                           linearDamping=0,
                                           angularDamping=0)

  def _BuildMotorIdList(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name]
        for motor_name in self._GetMotorNames()
    ]

  def _CreateRackConstraint(self, init_position, init_orientation):
    """Create a constraint that keeps the chassis at a fixed frame.

    This frame is defined by init_position and init_orientation.

    Args:
      init_position: initial position of the fixed frame.
      init_orientation: initial orientation of the fixed frame in quaternion
        format [x,y,z,w].

    Returns:
      Return the constraint id.
    """
    fixed_constraint = self._pybullet_client.createConstraint(
        parentBodyUniqueId=self.quadruped,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=self._pybullet_client.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=init_position,
        childFrameOrientation=init_orientation)
    return fixed_constraint

  def IsObservationValid(self):
    """Whether the observation is valid for the current time step.

    In simulation, observations are always valid. In real hardware, it may not
    be valid from time to time when communication error happens between the
    Nvidia TX2 and the microcontroller.

    Returns:
      Whether the observation is valid for the current time step.
    """
    return True

  def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    """Reset the minitaur to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the minitaur back to its starting position.
      default_motor_angles: The default motor angles. If it is None, minitaur
        will hold a default pose (motor angle math.pi / 2) for 100 steps. In
        torque control mode, the phase of holding the default pose is skipped.
      reset_time: The duration (in seconds) to hold the default motor angles. If
        reset_time <= 0 or in torque control mode, the phase of holding the
        default pose is skipped.
    """
    if reload_urdf:
      self._LoadRobotURDF()
      if self._on_rack:
        self.rack_constraint = (self._CreateRackConstraint(
            self._GetDefaultInitPosition(), self._GetDefaultInitOrientation()))
      self._BuildJointNameToIdDict()
      self._BuildUrdfIds()
      self._RemoveDefaultJointDamping()
      self._BuildMotorIdList()
      self._RecordMassInfoFromURDF()
      self._RecordInertiaInfoFromURDF()
      self.ResetPose(add_constraint=True)
    else:
      self._pybullet_client.resetBasePositionAndOrientation(
          self.quadruped, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0],
                                              [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors
    self._observation_history.clear()
    self._step_counter = 0
    self._state_action_counter = 0
    self._is_safe = True
    self._last_action = None
    self._SettleDownForReset(default_motor_angles, reset_time)
    if self._enable_action_filter:
      self._ResetActionFilter()
    if self._episode_history.curr_pointer>0:
      print("Saved", self._episode_history.curr_pointer)
      self._episode_history.done=True
      self._episode_history.save_to_file("readings.txt")
    else:
      print("not Saved")

  def _LoadRobotURDF(self):
    """Loads the URDF file for the robot."""
    urdf_file = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          urdf_file,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          urdf_file, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    """Sets the default motor angles and waits for the robot to settle down.

    The reset is skipped is reset_time is less than zereo.

    Args:
      default_motor_angles: A list of motor angles that the robot will achieve
        at the end of the reset phase.
      reset_time: The time duration for the reset phase.
    """
    self.ReceiveObservation()

    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)
    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)

  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=motor_id,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        force=torque)

  def _SetMotorTorqueByIds(self, motor_ids, torques):
    self._pybullet_client.setJointMotorControlArray(
        bodyIndex=self.quadruped,
        jointIndices=motor_ids,
        controlMode=self._pybullet_client.TORQUE_CONTROL,
        forces=torques)

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name],
                                   desired_angle)

  def GetURDFFile(self):
    return URDF_FILENAME

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    angles = self.GetDefaultInitJointPose()
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
        angle = angles[i]
        self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)

  def _ResetPoseForLeg(self, leg_id, add_constraint):
    """Reset the initial pose for the leg.

    Args:
      leg_id: It should be 0, 1, 2, or 3, which represents the leg at
        front_left, back_left, front_right and back_right.
      add_constraint: Whether to add a constraint at the joints of two feet.
    """
    knee_friction_force = 0
    half_pi = math.pi / 2.0
    knee_angle = -2.1834

    leg_position = LEG_POSITION[leg_id]
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "L_joint"],
        self._motor_direction[2 * leg_id] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "L_link"],
        self._motor_direction[2 * leg_id] * knee_angle,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["motor_" + leg_position + "R_joint"],
        self._motor_direction[2 * leg_id + 1] * half_pi,
        targetVelocity=0)
    self._pybullet_client.resetJointState(
        self.quadruped,
        self._joint_name_to_id["knee_" + leg_position + "R_link"],
        self._motor_direction[2 * leg_id + 1] * knee_angle,
        targetVelocity=0)
    if add_constraint:
      self._pybullet_client.createConstraint(
          self.quadruped,
          self._joint_name_to_id["knee_" + leg_position + "R_link"],
          self.quadruped,
          self._joint_name_to_id["knee_" + leg_position + "L_link"],
          self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
          KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

    # Disable the default motor in pybullet.
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["motor_" + leg_position +
                                           "L_joint"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["motor_" + leg_position +
                                           "R_joint"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)
    self._pybullet_client.setJointMotorControl2(
        bodyIndex=self.quadruped,
        jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
        controlMode=self._pybullet_client.VELOCITY_CONTROL,
        targetVelocity=0,
        force=knee_friction_force)

  def GetBasePosition(self):
    """Get the position of minitaur's base.

    Returns:
      The position of minitaur's base.
    """
    return self._base_position

  def GetBaseVelocity(self):
    """Get the linear velocity of minitaur's base.

    Returns:
      The velocity of minitaur's base.
    """
    velocity, _ = self._pybullet_client.getBaseVelocity(self.quadruped)
    return velocity

  def GetTrueBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame.
    """
    orientation = self.GetTrueBaseOrientation()
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
    return np.asarray(roll_pitch_yaw)

  def GetBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
      and latency.
    """
    delayed_orientation = np.array(
        self._control_observation[3 * self.num_motors:3 * self.num_motors + 4])
    delayed_roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(
        delayed_orientation)
    roll_pitch_yaw = self._AddSensorNoise(np.array(delayed_roll_pitch_yaw),
                                          self._observation_noise_stdev[3])
    return roll_pitch_yaw

  # def GetHipPositionsInBaseFrame(self):
  #   """Get the hip joint positions of the robot within its base frame."""
  #   return _DEFAULT_HIP_POSITIONS

  def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                              foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.

    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """
    assert len(self._foot_link_ids) == self.num_legs
    toe_id = self._foot_link_ids[leg_id]

    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = list(
        range(leg_id * motors_per_leg,
              leg_id * motors_per_leg + motors_per_leg))

    joint_angles = kinematics.joint_angles_from_link_position(
        robot=self,
        link_position=foot_local_position,
        link_id=toe_id,
        joint_ids=joint_position_idxs,
    )

    # Joint offset is necessary for Laikago.
    joint_angles = np.multiply(
        np.asarray(joint_angles) -
        np.asarray(self._motor_offset)[joint_position_idxs],
        self._motor_direction[joint_position_idxs])

    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()

  # def ComputeJacobian(self, leg_id):
  #   """Compute the Jacobian for a given leg."""
  #   # Does not work for Minitaur which has the four bar mechanism for now.
  #   assert len(self._foot_link_ids) == self.num_legs
  #   full_jacobian = kinematics.compute_jacobian(
  #       robot=self,
  #       link_id=self._foot_link_ids[leg_id],
  #   )
  #   motors_per_leg = self.num_motors // self.num_legs
  #   com_dof = 6
  #   return full_jacobian[com_dof + leg_id * motors_per_leg:com_dof +
  #                        (leg_id + 1) * motors_per_leg][(2, 0, 1), :]
  #
  # def MapContactForceToJointTorques(self, leg_id, contact_force):
  #   """Maps the foot contact force to the leg joint torques."""
  #   jv = self.ComputeJacobian(leg_id)
  #   motor_torques_list = np.matmul(contact_force, jv)
  #   motor_torques_dict = {}
  #   motors_per_leg = self.num_motors // self.num_legs
  #   for torque_id, joint_id in enumerate(
  #       range(leg_id * motors_per_leg, (leg_id + 1) * motors_per_leg)):
  #     motor_torques_dict[joint_id] = motor_torques_list[torque_id]
  #   return motor_torques_dict

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue

    return contacts

  def GetFootPositionsInBaseFrame(self):
    """Get the robot's foot position in the base frame."""
    assert len(self._foot_link_ids) == self.num_legs
    foot_positions = []
    for foot_id in self.GetFootLinkIDs():
      foot_positions.append(
          kinematics.link_position_in_base_frame(
              robot=self,
              link_id=foot_id,
          ))
    return np.array(foot_positions)

  def GetTrueMotorAngles(self):
    """Gets the eight motor angles at the current moment, mapped to [-pi, pi].

    Returns:
      Motor angles, mapped to [-pi, pi].
    """
    motor_angles = [state[0] for state in self._joint_states]
    motor_angles = np.multiply(
        np.asarray(motor_angles) - np.asarray(self._motor_offset),
        self._motor_direction)
    return motor_angles

  def GetMotorAngles(self):
    """Gets the eight motor angles.

    This function mimicks the noisy sensor reading and adds latency. The motor
    angles that are delayed, noise polluted, and mapped to [-pi, pi].

    Returns:
      Motor angles polluted by noise and latency, mapped to [-pi, pi].
    """
    motor_angles = self._AddSensorNoise(
        np.array(self._control_observation[0:self.num_motors]),
        self._observation_noise_stdev[0])
    return MapToMinusPiToPi(motor_angles)

  def GetTrueMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [state[1] for state in self._joint_states]

    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorVelocities(self):
    """Get the velocity of all eight motors.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Velocities of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[self.num_motors:2 *
                                           self.num_motors]),
        self._observation_noise_stdev[1])

  def GetTrueMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    Returns:
      Motor torques of all eight motors.
    """
    return self._observed_motor_torques

  def GetMotorTorques(self):
    """Get the amount of torque the motors are exerting.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Motor torques of all eight motors polluted by noise and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[2 * self.num_motors:3 *
                                           self.num_motors]),
        self._observation_noise_stdev[2])

  def GetEnergyConsumptionPerControlStep(self):
    """Get the amount of energy used in last one time step.

    Returns:
      Energy Consumption based on motor velocities and torques (Nm^2/s).
    """
    return np.abs(np.dot(
        self.GetMotorTorques(),
        self.GetMotorVelocities())) * self.time_step * self._action_repeat

  def GetTrueBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    Returns:
      The orientation of minitaur's base.
    """
    return self._base_orientation

  def GetBaseOrientation(self):
    """Get the orientation of minitaur's base, represented as quaternion.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      The orientation of minitaur's base polluted by noise and latency.
    """
    return self._pybullet_client.getQuaternionFromEuler(
        self.GetBaseRollPitchYaw())

  def GetTrueBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base.
    """
    angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
    orientation = self.GetTrueBaseOrientation()
    return self.TransformAngularVelocityToLocalFrame(angular_velocity,
                                                     orientation)

  def TransformAngularVelocityToLocalFrame(self, angular_velocity,
                                           orientation):
    """Transform the angular velocity from world frame to robot's frame.

    Args:
      angular_velocity: Angular velocity of the robot in world frame.
      orientation: Orientation of the robot represented as a quaternion.

    Returns:
      angular velocity of based on the given orientation.
    """
    # Treat angular velocity as a position vector, then transform based on the
    # orientation given by dividing (or multiplying with inverse).
    # Get inverse quaternion assuming the vector is at 0,0,0 origin.
    _, orientation_inversed = self._pybullet_client.invertTransform(
        [0, 0, 0], orientation)
    # Transform the angular_velocity at neutral orientation using a neutral
    # translation and reverse of the given orientation.
    relative_velocity, _ = self._pybullet_client.multiplyTransforms(
        [0, 0, 0], orientation_inversed, angular_velocity,
        self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
    return np.asarray(relative_velocity)

  def GetBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base polluted by noise
      and latency.
    """
    return self._AddSensorNoise(
        np.array(self._control_observation[3 * self.num_motors +
                                           4:3 * self.num_motors + 7]),
        self._observation_noise_stdev[4])

  def GetActionDimension(self):
    """Get the length of the action list.

    Returns:
      The length of the action list.
    """
    return self.num_motors

  def _ApplyOverheatProtection(self, actual_torque):
    if self._motor_overheat_protection:
      for i in range(self.num_motors):
        if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
          self._overheat_counter[i] += 1
        else:
          self._overheat_counter[i] = 0
        if (self._overheat_counter[i] >
            OVERHEAT_SHUTDOWN_TIME / self.time_step):
          self._motor_enabled_list[i] = False

  def ApplyAction(self, motor_commands, motor_control_mode):
    """Apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)
    self.last_action_time = self._state_action_counter * self.time_step
    control_mode = motor_control_mode

    if control_mode is None:
      control_mode = self._motor_control_mode

    motor_commands = np.asarray(motor_commands)

    q, qdot = self._GetPDObservation()
    qdot_true = self.GetTrueMotorVelocities()
    actual_torque, observed_torque = self._motor_model.convert_to_torque(
        motor_commands, q, qdot, qdot_true, control_mode)

    # May turn off the motor
    self._ApplyOverheatProtection(actual_torque)

    # The torque is already in the observation space because we use
    # GetMotorAngles and GetMotorVelocities.
    self._observed_motor_torques = observed_torque

    # Transform into the motor space when applying the torque.
    self._applied_motor_torque = np.multiply(actual_torque,
                                             self._motor_direction)
    motor_ids = []
    motor_torques = []

    for motor_id, motor_torque, motor_enabled in zip(
        self._motor_id_list, self._applied_motor_torque,
        self._motor_enabled_list):
      if motor_enabled:
        motor_ids.append(motor_id)
        motor_torques.append(motor_torque)
      else:
        motor_ids.append(motor_id)
        motor_torques.append(0)
    self._SetMotorTorqueByIds(motor_ids, motor_torques)

  def ConvertFromLegModel(self, actions):
    """Convert the actions that use leg model to the real motor actions.

    Args:
      actions: The theta, phi of the leg model.

    Returns:
      The eight desired motor angles that can be used in ApplyActions().
    """
    motor_angle = copy.deepcopy(actions)
    scale_for_singularity = 1
    offset_for_singularity = 1.5
    half_num_motors = self.num_motors // 2
    quater_pi = math.pi / 4
    for i in range(self.num_motors):
      action_idx = i // 2
      forward_backward_component = (
          -scale_for_singularity * quater_pi *
          (actions[action_idx + half_num_motors] + offset_for_singularity))
      extension_component = (-1)**i * quater_pi * actions[action_idx]
      if i >= half_num_motors:
        extension_component = -extension_component
      motor_angle[i] = (math.pi + forward_backward_component +
                        extension_component)
    return motor_angle

  def GetBaseMassesFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetBaseInertiasFromURDF(self):
    """Get the inertia of the base from the URDF file."""
    return self._base_inertia_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetLegInertiasFromURDF(self):
    """Get the inertia of the legs from the URDF file."""
    return self._leg_inertia_urdf

  def SetBaseMasses(self, base_mass):
    """Set the mass of minitaur's base.

    Args:
      base_mass: A list of masses of each body link in CHASIS_LINK_IDS. The
        length of this list should be the same as the length of CHASIS_LINK_IDS.

    Raises:
      ValueError: It is raised when the length of base_mass is not the same as
        the length of self._chassis_link_ids.
    """
    if len(base_mass) != len(self._chassis_link_ids):
      raise ValueError(
          "The length of base_mass {} and self._chassis_link_ids {} are not "
          "the same.".format(len(base_mass), len(self._chassis_link_ids)))
    for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
      self._pybullet_client.changeDynamics(self.quadruped,
                                           chassis_id,
                                           mass=chassis_mass)

  def SetLegMasses(self, leg_masses):
    """Set the mass of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link masses, last 8 correspond
    to motor masses (24 total).

    Args:
      leg_masses: The leg and motor masses for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of masses is not equal to number
        of links + motors.
    """
    if len(leg_masses) != len(self._leg_link_ids) + len(self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
      self._pybullet_client.changeDynamics(self.quadruped,
                                           leg_id,
                                           mass=leg_mass)
    motor_masses = leg_masses[len(self._leg_link_ids):]
    for link_id, motor_mass in zip(self._motor_link_ids, motor_masses):
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           mass=motor_mass)

  def SetBaseInertias(self, base_inertias):
    """Set the inertias of minitaur's base.

    Args:
      base_inertias: A list of inertias of each body link in CHASIS_LINK_IDS.
        The length of this list should be the same as the length of
        CHASIS_LINK_IDS.

    Raises:
      ValueError: It is raised when the length of base_inertias is not the same
        as the length of self._chassis_link_ids and base_inertias contains
        negative values.
    """
    if len(base_inertias) != len(self._chassis_link_ids):
      raise ValueError(
          "The length of base_inertias {} and self._chassis_link_ids {} are "
          "not the same.".format(len(base_inertias),
                                 len(self._chassis_link_ids)))
    for chassis_id, chassis_inertia in zip(self._chassis_link_ids,
                                           base_inertias):
      for inertia_value in chassis_inertia:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(
          self.quadruped, chassis_id, localInertiaDiagonal=chassis_inertia)

  def SetLegInertias(self, leg_inertias):
    """Set the inertias of the legs.

    A leg includes leg_link and motor. 4 legs contain 16 links (4 links each)
    and 8 motors. First 16 numbers correspond to link inertia, last 8 correspond
    to motor inertia (24 total).

    Args:
      leg_inertias: The leg and motor inertias for all the leg links and motors.

    Raises:
      ValueError: It is raised when the length of inertias is not equal to
      the number of links + motors or leg_inertias contains negative values.
    """

    if len(leg_inertias) != len(self._leg_link_ids) + len(
        self._motor_link_ids):
      raise ValueError("The number of values passed to SetLegMasses are "
                       "different than number of leg links and motors.")
    for leg_id, leg_inertia in zip(self._leg_link_ids, leg_inertias):
      for inertia_value in leg_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           leg_id,
                                           localInertiaDiagonal=leg_inertia)

    motor_inertias = leg_inertias[len(self._leg_link_ids):]
    for link_id, motor_inertia in zip(self._motor_link_ids, motor_inertias):
      for inertia_value in motor_inertias:
        if (np.asarray(inertia_value) < 0).any():
          raise ValueError("Values in inertia matrix should be non-negative.")
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           localInertiaDiagonal=motor_inertia)

  def SetFootFriction(self, foot_friction):
    """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           lateralFriction=foot_friction)

  def SetFootRestitution(self, foot_restitution):
    """Set the coefficient of restitution at the feet.

    Args:
      foot_restitution: The coefficient of restitution (bounciness) of the feet.
        This value is shared by all four feet.
    """
    for link_id in self._foot_link_ids:
      self._pybullet_client.changeDynamics(self.quadruped,
                                           link_id,
                                           restitution=foot_restitution)

  def SetJointFriction(self, joint_frictions):
    for knee_joint_id, friction in zip(self._foot_link_ids, joint_frictions):
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=knee_joint_id,
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=friction)

  def GetNumKneeJoints(self):
    return len(self._foot_link_ids)

  def SetBatteryVoltage(self, voltage):
    self._motor_model.set_voltage(voltage)

  def SetMotorViscousDamping(self, viscous_damping):
    self._motor_model.set_viscous_damping(viscous_damping)

  def GetTrueObservation(self):
    observation = []
    observation.extend(self.GetTrueMotorAngles())
    observation.extend(self.GetTrueMotorVelocities())
    observation.extend(self.GetTrueMotorTorques())
    observation.extend(self.GetTrueBaseOrientation())
    observation.extend(self.GetTrueBaseRollPitchYawRate())
    return observation

  def ReceiveObservation(self):
    """Receive the observation from sensors.

    This function is called once per step. The observations are only updated
    when this function is called.
    """
    self._joint_states = self._pybullet_client.getJointStates(
        self.quadruped, self._motor_id_list)
    self._base_position, orientation = (
        self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    # Computes the relative orientation relative to the robot's
    # initial_orientation.
    _, self._base_orientation = self._pybullet_client.multiplyTransforms(
        positionA=[0, 0, 0],
        orientationA=orientation,
        positionB=[0, 0, 0],
        orientationB=self._init_orientation_inv)
    self._observation_history.appendleft(self.GetTrueObservation())
    self._control_observation = self._GetControlObservation()
    self.last_state_time = self._state_action_counter * self.time_step

  def _GetDelayedObservation(self, latency):
    """Get observation that is delayed by the amount specified in latency.

    Args:
      latency: The latency (in seconds) of the delayed observation.

    Returns:
      observation: The observation which was actually latency seconds ago.
    """
    if latency <= 0 or len(self._observation_history) == 1:
      observation = self._observation_history[0]
    else:
      n_steps_ago = int(latency / self.time_step)
      if n_steps_ago + 1 >= len(self._observation_history):
        return self._observation_history[-1]
      remaining_latency = latency - n_steps_ago * self.time_step
      blend_alpha = remaining_latency / self.time_step
      observation = (
          (1.0 - blend_alpha) *
          np.array(self._observation_history[n_steps_ago]) +
          blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
    return observation

  def _GetPDObservation(self):
    pd_delayed_observation = self._GetDelayedObservation(self._pd_latency)
    q = pd_delayed_observation[0:self.num_motors]
    qdot = pd_delayed_observation[self.num_motors:2 * self.num_motors]
    return (np.array(q), np.array(qdot))

  def _GetControlObservation(self):
    control_delayed_observation = self._GetDelayedObservation(
        self._control_latency)
    return control_delayed_observation

  def _AddSensorNoise(self, sensor_values, noise_stdev):
    if noise_stdev <= 0:
      return sensor_values
    observation = sensor_values + np.random.normal(scale=noise_stdev,
                                                   size=sensor_values.shape)
    return observation

  def SetControlLatency(self, latency):
    """Set the latency of the control loop.

    It measures the duration between sending an action from Nvidia TX2 and
    receiving the observation from microcontroller.

    Args:
      latency: The latency (in seconds) of the control loop.
    """
    self._control_latency = latency

  def GetControlLatency(self):
    """Get the control latency.

    Returns:
      The latency (in seconds) between when the motor command is sent and when
        the sensor measurements are reported back to the controller.
    """
    return self._control_latency

  def SetMotorGains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain(s) of the motors.
      kd: derivative gain(s) of the motors.
    """
    if isinstance(kp, (collections.Sequence, np.ndarray)):
      self._motor_kps = np.asarray(kp)
    else:
      self._motor_kps = np.full(self.num_motors, kp)

    if isinstance(kd, (collections.Sequence, np.ndarray)):
      self._motor_kds = np.asarray(kd)
    else:
      self._motor_kds = np.full(self.num_motors, kd)

    self._motor_model.set_motor_gains(kp, kd)

  def GetMotorGains(self):
    """Get the gains of the motor.

    Returns:
      The proportional gain.
      The derivative gain.
    """
    return self._motor_kps, self._motor_kds

  def GetMotorPositionGains(self):
    """Get the position gains of the motor.

    Returns:
      The proportional gain.
    """
    return self._motor_kps

  def GetMotorVelocityGains(self):
    """Get the velocity gains of the motor.

    Returns:
      The derivative gain.
    """
    return self._motor_kds

  def SetMotorStrengthRatio(self, ratio):
    """Set the strength of all motors relative to the default value.

    Args:
      ratio: The relative strength. A scalar range from 0.0 to 1.0.
    """
    self._motor_model.set_strength_ratios([ratio] * self.num_motors)

  def SetMotorStrengthRatios(self, ratios):
    """Set the strength of each motor relative to the default value.

    Args:
      ratios: The relative strength. A numpy array ranging from 0.0 to 1.0.
    """
    self._motor_model.set_strength_ratios(ratios)

  def SetTimeSteps(self, action_repeat, simulation_step):
    """Set the time steps of the control and simulation.

    Args:
      action_repeat: The number of simulation steps that the same action is
        repeated.
      simulation_step: The simulation time step.
    """
    self.time_step = simulation_step
    self._action_repeat = action_repeat

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    # The Laikago URDF assumes the initial pose of heading towards z axis,
    # and belly towards y axis. The following transformation is to transform
    # the Laikago initial orientation to our commonly used orientation: heading
    # towards -x direction, and z axis is the up direction.
    init_orientation = pyb.getQuaternionFromEuler(
        [math.pi / 2.0, 0, math.pi / 2.0])
    return init_orientation

  @property
  def chassis_link_ids(self):
    return self._chassis_link_ids

  def SetAllSensors(self, sensors):
    """set all sensors to this robot and move the ownership to this robot.

    Args:
      sensors: a list of sensors to this robot.
    """
    for s in sensors:
      s.set_robot(self)
    self._sensors = sensors

  def GetAllSensors(self):
    """get all sensors associated with this robot.

    Returns:
      sensors: a list of all sensors.
    """
    return self._sensors

  def GetSensor(self, name):
    """get the first sensor with the given name.

    This function return None if a sensor with the given name does not exist.

    Args:
      name: the name of the sensor we are looking

    Returns:
      sensor: a sensor with the given name. None if not exists.
    """
    for s in self._sensors:
      if s.get_name() == name:
        return s
    return None

  @property
  def is_safe(self):
    return self._is_safe

  @property
  def last_action(self):
    return self._last_action

  def ProcessAction(self, action, substep_count):
    """If enabled, interpolates between the current and previous actions.

    Args:
      action: current action.
      substep_count: the step count should be between [0, self.__action_repeat).

    Returns:
      If interpolation is enabled, returns interpolated action depending on
      the current action repeat substep.
    """
    if self._enable_action_interpolation and self._last_action is not None:
      lerp = float(substep_count + 1) / self._action_repeat
      proc_action = self._last_action + lerp * (action - self._last_action)
    else:
      proc_action = action

    return proc_action

  def _BuildActionFilter(self):
    sampling_rate = 1 / (self.time_step * self._action_repeat)
    num_joints = self.GetActionDimension()
    a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                num_joints=num_joints)
    return a_filter

  def _ResetActionFilter(self):
    self._action_filter.reset()

  def _FilterAction(self, action):
    # initialize the filter history, since resetting the filter will fill
    # the history with zeros and this can cause sudden movements at the start
    # of each episode
    if self._step_counter == 0:
      default_action = self.GetMotorAngles()
      self._action_filter.init_history(default_action)

    filtered_action = self._action_filter.filter(action)
    return filtered_action


  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._knee_link_ids = []
    self._foot_link_ids = []

    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if _CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_ids.append(joint_id)
      elif _MOTOR_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif _KNEE_NAME_PATTERN.match(joint_name):
        self._knee_link_ids.append(joint_id)
      elif _TOE_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._knee_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)
    if self._allow_knee_contact:
      self._foot_link_ids.extend(self._knee_link_ids)

    self._chassis_link_ids.sort()
    self._motor_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose

  def GetInitMotorAngles(self):
      return INIT_MOTOR_ANGLES

  def GetActionLimit(self):
      return np.array([2] * NUM_MOTORS)

  def _ClipMotorCommands(self, motor_commands):
    """Clips motor commands.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).

    Returns:
      Clipped motor commands.
    """

    # clamp the motor command by the joint limit, in case weired things happens
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

  @classmethod
  def GetConstants(cls):
    del cls
    return laikago_constants

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def joint_states(self):
    return self._joint_states
