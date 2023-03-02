import numpy as np


# Convention for storing the info
# Robot:
# Frequency?:
# Joints:
# Stiffness:
# Damping:


# Data -->
# Joints, Actuators

class HistoryStepsAPI:
    def __init__(self, robot_name, joint_names, stiffness=[], damping=[], freq=30) -> None:
        self._robot_name = robot_name
        self._joint_names = joint_names
        self._stiffness = stiffness if len(stiffness)>0 else np.ones(len(joint_names))*100
        self._damping = damping if len(damping)>0 else np.ones(len(joint_names))
        self.curr_pointer=0
        self.history=None
        self.idx_num = len(joint_names)
        self.begin_readings()
        self.done = False
    
    def begin_readings(self, max_length=900):
        self.history = np.zeros((max_length, len(self._joint_names)*3))
        self.curr_pointer=0
        self._max_length=max_length

    def take_reading(self, joint_obs, motor_torque, desired_pd):
        if self.done: return 
        print(self.curr_pointer)
        assert len(joint_obs) == len(self._joint_names)
        assert len(motor_torque) == len(self._joint_names)
        if self.curr_pointer >= self._max_length:
            pass
        else:
            self.history[self.curr_pointer, :self.idx_num] = joint_obs
            self.history[self.curr_pointer, self.idx_num:2*self.idx_num] = motor_torque
            self.history[self.curr_pointer, 2*self.idx_num:] = desired_pd
            self.curr_pointer+= 1

    def save_to_file(self, out_filename):
        comments = "\n"
        comments+=(f'Robot: {self._robot_name} \n')
        comments+=(f'Frequency: ? \n')
        comments+=(f'Joints {self._joint_names} \n')
        comments+=(f'Stiffness: {self._stiffness} \n')
        comments+=(f'Damping: {self._damping} \n')
        comments+=(f'Episode Nume Control Frames: {self.curr_pointer} \n')
        comments+=("[Angle Observation, Motor Torque, Desired Joint Position (Post-Action FIlter and clipping)] \n")
        np.savetxt(out_filename, self.history, header=comments)
    def load_from_file(self, in_filename):
        self.history = np.loadtxt(in_filename)


    def next(self):
        if self.curr_pointer>= len(self.history):
            return None, None
        out = self.history[self.curr_pointer, :]
        self.curr_pointer+=1 
        return out[:self.idx_num], out[self.idx_num:2*self.idx_num], out[2*self.idx_num:] 
    