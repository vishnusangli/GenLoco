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

NUM_MOTORS = 10
NUM_LEGS = 2
DOFS_PER_LEG = 5
indices = [0, 1, 2, 3, 6, 7, 8, 9, 10, 13]
MOTOR_NAMES = ['hip_abduction_left', #0
	       'hip_rotation_left', #1 
		   'hip_flexion_left', #2
		   'knee_joint_left', #3
		   'toe_joint_left', #4
		   'hip_abduction_right', #5 
		   'hip_rotation_right', #6
		   'hip_flexion_right', #7
		   'knee_joint_right', #8
		   'toe_joint_right', #9
		   ]
FULL_MOTOR_NAMES = ['hip_abduction_left', #0
	       'hip_rotation_left', #1 
		   'hip_flexion_left', #2
		   'knee_joint_left', #3
		   'toe_joint_left', #4
		   'hip_abduction_right', #5 
		   'hip_rotation_right', #6
		   'hip_flexion_right', #7
		   'knee_joint_right', #8
		   'toe_joint_right', #9
		   'knee_to_shin_left', #give 0 
		   'ankle_joint_left', #3
		   'knee_to_shin_right', #give 0 
		   'ankle_joint_right', #8
		   ]

HIP_NAME_PATTERN = re.compile(r"hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"knee_\w+")
LOWER_NAME_PATTERN = re.compile(r"ankle_\w+")
TOE_NAME_PATTERN = re.compile(r"toe_\w*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

_DEFAULT_TORQUE_LIMITS = [30] * NUM_MOTORS
INIT_POSITION = [0, 0, 0.8]
JOINT_DIRECTIONS = np.array([-1]*NUM_MOTORS)

JOINT_OFFSETS = np.array([0.0] *NUM_MOTORS)

# Bases on the readings from 's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0, 1.0204, -1.97, -1.9, 0, 0, 1.0204, -1.97, -1.9])
FULL_INIT_MOTOR_ANGLES = np.array([0,0,1.0204,-1.97,-1.9,0,0,1.0204,-1.97,-1.9, -0.084,2.06, -0.084,2.06])
"""
10:
np.array([-0.2618, -0.3927, -0.8727, -2.8623, -2.4435, -0.3927, -0.3927, -0.8727, -2.8623, -2.4435])
np.array([0.3927, 0.3927, 1.3963, -0.6458, -0.5236, 0.2618, 0.3927, 1.3963, -0.6458, -0.5236])

14:
np.array([-0.2618, -0.3927, -0.8727, -2.8623, -0.349, 0.872, -2.4435, -0.3927, -0.3927, -0.8727, -2.8623, -0.349, 0.872, -2.4435])
np.array([0.3927, 0.3927, 1.3963, -0.6458, 0.349, 2.97, -0.5236, 0.2618, 0.3927, 1.3963, -0.6458, 0.349, 2.97, -0.5236])
"""
MOTOR_LOWER_BOUNDS = np.array([-0.2618, -0.3927, -0.8727, -2.8623, -2.4435, -0.3927, -0.3927, -0.8727, -2.8623, -2.4435])
MOTOR_UPPER_BOUNDS = np.array([0.3927, 0.3927, 1.3963, -0.6458, -0.5236, 0.2618, 0.3927, 1.3963, -0.6458, -0.5236])


def ImputePDAangles(pose, a=0.0, b=lambda x: np.radians(13)-x):
	altered_pose = np.concatenate([pose[:4], [a, b(pose[4])], pose[4:9], [a, b(pose[8])], pose[9:]])
	return altered_pose
def ReducePDAngles(pose):
	altered_pose = pose[indices]
	return altered_pose

ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(MOTOR_NAMES,
                                        upper_bound=MOTOR_UPPER_BOUNDS[i],
                                        lower_bound=MOTOR_LOWER_BOUNDS[i]) 
										for i in range(NUM_MOTORS)
										]

pGain = np.array([400, 200, 200, 500, 20, 400, 200, 200, 500, 20]) 
dGain = np.array([4, 4, 10, 20, 4, 4, 4, 10, 20, 4])

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
		motor_kp =  [100]*NUM_MOTORS #pGain
		motor_kd = [1]*NUM_MOTORS #dGain
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
		self._BuildFullMotorIdList()

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
		angles = FULL_INIT_MOTOR_ANGLES
		for name, i in zip(FULL_MOTOR_NAMES, range(len(FULL_MOTOR_NAMES))):
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
		contacts = [False, False]
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
	
	def GetActionLimit(self):
		return np.array([2] * NUM_MOTORS)
	
	def _BuildFullMotorIdList(self):
		self._full_motor_id_list = [
			self._joint_name_to_id[motor_name]
			for motor_name in FULL_MOTOR_NAMES
		]
	
	def GetTrueMotorAngles(self):
		"""Gets the eight motor angles at the current moment, mapped to [-pi, pi].

		Returns:
		Motor angles, mapped to [-pi, pi].
		"""
		motor_angles = [state[0] for state in self._joint_states]
		motor_angles = np.multiply(
			np.asarray(motor_angles) - np.asarray(JOINT_OFFSETS),
			JOINT_DIRECTIONS)
		return motor_angles
	
	def GetTrueMotorVelocities(self):
		"""Get the velocity of all eight motors.

		Returns:
		Velocities of all eight motors.
		"""
		motor_velocities = [state[1] for state in self._joint_states]

		motor_velocities = np.multiply(motor_velocities, JOINT_DIRECTIONS)
		return motor_velocities

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

	def _SetMotorTorqueByIds(self, motor_ids, torques):
		new_torques = np.concatenate([torques[:4], [0.0, -torques[3]], torques[4: 9], 
									[0., -torques[8]], torques[9:]])
		self._pybullet_client.setJointMotorControlArray(
			bodyIndex=self.quadruped,
			jointIndices=self._full_motor_id_list,
			controlMode=self._pybullet_client.TORQUE_CONTROL,
			forces=new_torques)
