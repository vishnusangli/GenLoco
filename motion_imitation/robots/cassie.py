################# REAL CODE NOW
import math
import os
# import gin
import numpy as np
import re
from motion_imitation.robots import base_robot
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.3
URDF_FILENAME = "robot_descriptions/cassie_description/urdf/cassie_collide.urdf"

NUM_MOTORS = 14
NUM_LEGS = 2
DOFS_PER_LEG = 7

MOTOR_NAMES = ['hip_abduction_left', 
	       'hip_rotation_left', 
		   'hip_flexion_left', 
		   'knee_joint_left', 
		   'knee_to_shin_left', 
		   'ankle_joint_left', 
		   'toe_joint_left', 
		   'hip_abduction_right', 
		   'hip_rotation_right', 
		   'hip_flexion_right', 
		   'knee_joint_right', 
		   'knee_to_shin_right', 
		   'ankle_joint_right', 
		   'toe_joint_right']

HIP_NAME_PATTERN = re.compile(r"hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"knee_\w+")
LOWER_NAME_PATTERN = re.compile(r"ankle_\w+")
TOE_NAME_PATTERN = re.compile(r"toe_\w*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

_DEFAULT_TORQUE_LIMITS = [50] * NUM_MOTORS
INIT_POSITION = [0, 0, 0.8]
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

JOINT_OFFSETS = np.array([0.0] *NUM_MOTORS)
PI = math.pi

# Bases on the readings from 's default pose.
INIT_MOTOR_ANGLES = np.array([0,0,1.0204,-1.97,-0.084,2.06,-1.9,0,0,1.0204,-1.97,-0.084,2.06,-1.9])
#self.action_bounds = np.array([[-0.2618, -0.3927, -0.8727, -2.8623, -2.4435, -0.3927, -0.3927, -0.8727, -2.8623, -2.4435],                              [0.3927, 0.3927, 1.3963, -0.6458, -0.5236, 0.2618, 0.3927, 1.3963, -0.6458, -0.5236]])
MOTOR_LOWER_BOUNDS = np.array([-0.2618, -0.3927, -0.8727, -2.8623, -0.349, 0.872, -2.4435, -0.3927, -0.3927, -0.8727, -2.8623, -0.349, 0.872, -2.4435])
#4, 5, 11, 12
MOTOR_UPPER_BOUNDS = np.array([0.3927, 0.3927, 1.3963, -0.6458, 0.349, 2.97, -0.5236, 0.2618, 0.3927, 1.3963, -0.6458, 0.349, 2.97, -0.5236])

ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name=MOTOR_NAMES[i],
                                        upper_bound=MOTOR_UPPER_BOUNDS[i],
                                        lower_bound=-MOTOR_LOWER_BOUNDS[i]) 
										for i in range(len(MOTOR_NAMES))
										]
P_GAIN = 100.0
D_GAIN = 1.

motor_kp = []
motor_kd = []

class Cassie(base_robot.Base_robot):
	"""A simulation for the anymal robot."""
	NUM_LEGS=NUM_LEGS
	NUM_MOTORS = NUM_MOTORS
	ACTION_CONFIG = ACTION_CONFIG
	def __init__(self,
	             pybullet_client,
	             motor_control_mode,
	             urdf_filename=URDF_FILENAME,
	             enable_clip_motor_commands=False,
	             time_step=0.001,
	             action_repeat=33,
	             sensors=None,
	             control_latency=0.002,
	             on_rack=False,
	             enable_action_interpolation=True,
	             enable_action_filter=False,
	             reset_time=-1,
	             allow_knee_contact=False,
	             ):

		self._urdf_filename = urdf_filename
		self._allow_knee_contact = allow_knee_contact
		self._enable_clip_motor_commands = enable_clip_motor_commands
		motor_kp = [P_GAIN] * NUM_MOTORS
		motor_kd = [D_GAIN] *NUM_MOTORS
		super(Cassie, self).__init__(
			pybullet_client=pybullet_client,
			num_motors=NUM_MOTORS,
			dofs_per_leg=DOFS_PER_LEG,
			time_step=time_step,
			action_repeat=action_repeat,
			motor_direction=JOINT_DIRECTIONS,
			motor_torque_limits=_DEFAULT_TORQUE_LIMITS,
			motor_offset=JOINT_OFFSETS,
			motor_overheat_protection=False,
			motor_control_mode=motor_control_mode,
			sensors=sensors,
			motor_kp=motor_kp,
			motor_kd=motor_kd,
			control_latency=control_latency,
			on_rack=on_rack,
			enable_action_interpolation=enable_action_interpolation,
			enable_action_filter=enable_action_filter,
			reset_time=reset_time,
			allow_knee_contact=allow_knee_contact,
			enable_clip_motor_commands=enable_clip_motor_commands)

	def GetURDFFile(self):
		return self._urdf_filename

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
			self._pybullet_client.resetJointState(
				self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

	def _BuildUrdfIds(self):
		num_joints = self.pybullet_client.getNumJoints(self.quadruped)
		self._hip_link_ids = [-1]
		self._leg_link_ids = []
		self._motor_link_ids = []
		self._lower_link_ids = []
		self._foot_link_ids = []
		self._imu_link_ids = []

		for i in range(num_joints):
			joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
			joint_name = joint_info[1].decode("UTF-8")
			joint_id = self._joint_name_to_id[joint_name]
			if HIP_NAME_PATTERN.match(joint_name):
				self._hip_link_ids.append(joint_id)
			elif UPPER_NAME_PATTERN.match(joint_name):
				self._motor_link_ids.append(joint_id)
			# We either treat the lower leg or the toe as the foot link, depending on
			# the urdf version used.HIHIP_NAME_PATTERNP_NAME_PATTERN
			elif LOWER_NAME_PATTERN.match(joint_name):
				self._lower_link_ids.append(joint_id)
			elif TOE_NAME_PATTERN.match(joint_name):
				#assert self._urdf_filename == URDF_WITH_TOES
				self._foot_link_ids.append(joint_id)
			elif IMU_NAME_PATTERN.match(joint_name):
				self._imu_link_ids.append(joint_id)
			# else:
			# 	raise ValueError("Unknown category of joint %s" % joint_name)
		self._leg_link_ids.extend(self._motor_link_ids)
		self._leg_link_ids.extend(self._lower_link_ids)
		if self._allow_knee_contact:
			self._foot_link_ids.extend(self._lower_link_ids)
			
		self._hip_link_ids.sort()
		self._motor_link_ids.sort()
		self._lower_link_ids.sort()
		self._foot_link_ids.sort()
		self._leg_link_ids.sort()

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
		
	def _GetMotorNames(self):
		return MOTOR_NAMES

	def _GetDefaultInitPosition(self):
		return INIT_POSITION

	def _GetDefaultInitOrientation(self):
		init_orientation = [0, 0, 0, 1.0]
		return init_orientation

	def GetDefaultInitJointPose(self):
		"""Get default initial joint pose."""
		joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
		return joint_pose

	def GetInitMotorAngles(self):
		return INIT_MOTOR_ANGLES

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

	def GetFootContacts(self): # check the id
		all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)
		contacts = [False]
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

	def GetFootLinkIDs(self):
		"""Get list of IDs for all foot links."""
		return self._foot_link_ids

	def GetInitMotorAngles(self):
		return INIT_MOTOR_ANGLES
	
	def GetActionLimit(self):
		return np.array([2] * NUM_MOTORS)