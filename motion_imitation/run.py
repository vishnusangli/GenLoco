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
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.robots import anymal_b_simple, anymal_c_simple,base_robot,mini_cheetah,go1,aliengo,spot,spotmicro,siriusmid_belt, cassie
from motion_imitation.robots import a1
from motion_imitation.real_a1 import a1_robot_real
from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True

robot_classes = {
    "laikago" : base_robot.Base_robot,
    "a1" : a1.A1,
    "anymal_b":anymal_b_simple.Anymal_b,
    "anymal_c":anymal_c_simple.Anymal_c,
    "siriusmid_belt":siriusmid_belt.Siriusmid_belt,
    "mini_cheetah":mini_cheetah.Mini_cheetah,
    "go1":go1.Go1,
    "aliengo":aliengo.Aliengo,
    "spot":spot.Spot,
    "spotmicro":spotmicro.SpotMicro,
    "real_a1":a1_robot_real.A1Robot,
    # add new robot class here
    "cassie": cassie.Cassie
}

def set_rand_seed(seed=None):
  if seed is None:
    seed = int(time.time())

  seed += 97 * MPI.COMM_WORLD.Get_rank()

  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  policy_kwargs = {
      "net_arch": [{"pi": [1024, 512],
                    "vf": [1024, 512]}],
      "act_fun": tf.nn.relu
  }

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=timesteps_per_actorbatch,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-4,
               optim_batchsize=optim_batchsize,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant',
               policy_kwargs=policy_kwargs,
               tensorboard_log=output_dir,
               verbose=1)
  return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0, use_curr=True):
  if (output_dir == ""):
    save_path = None
  else:
    save_path = os.path.join(output_dir, "model.zip")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  

  callbacks = []
  # Save a checkpoint every n steps
  if (output_dir != ""):
    if (int_save_freq > 0):
      int_dir = os.path.join(output_dir, "intermedate")
      callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                          name_prefix='model'))

  model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks, use_curriculum=use_curr)

  return

def test(model, env, num_procs, num_episodes=None):
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  o = env.reset()
  while episode_count < num_local_episodes:
    print(episode_count)
    a, _ = model.predict(o, deterministic=True)
    o, r, done, info = env.step(a)
    curr_return += r

    if done:
        o = env.reset()
        sum_return += curr_return
        episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  mean_return = sum_return / episode_count

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))

  return

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
  arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/a1_pace.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps
  arg_parser.add_argument("--robot", dest="robot", type=str, default="")
  arg_parser.add_argument("--timesteps_per_actorbatch", dest="timesteps_per_actorbatch", type=int, default=TIMESTEPS_PER_ACTORBATCH)
  arg_parser.add_argument("--optim_batchsize", dest="optim_batchsize", type=int, default=OPTIM_BATCHSIZE)
  arg_parser.add_argument("--phase_only", dest="phase_only", action="store_true", default=False)
  arg_parser.add_argument("--randomized_robot", dest="randomized_robot", action="store_true", default=False)
  arg_parser.add_argument("--sync_root_rotation", dest="sync_root_rotation", action="store_true", default=False)

  args = arg_parser.parse_args()

  if args.randomized_robot:
    robot_class = None
  else:
    robot_class = robot_classes[args.robot]
  
  num_procs = MPI.COMM_WORLD.Get_size()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
  env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_sync_root_rotation=args.sync_root_rotation,
                                        enable_rendering=args.visualize,
                                        enable_phase_only=args.phase_only,
                                        enable_randomized_robot=args.randomized_robot,
                                        robot_class=robot_class,
                                        visualize=args.visualize)
  
  model = build_model(env=env,
                      num_procs=num_procs,
                      timesteps_per_actorbatch=args.timesteps_per_actorbatch,
                      optim_batchsize=args.optim_batchsize,
                      output_dir=args.output_dir)

  use_curriculum=True
  if args.model_file != "":
    model.load_parameters(args.model_file)
    use_curriculum = False

  if args.mode == "train":
      train(model=model, 
            env=env, 
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            int_save_freq=args.int_save_freq,
            use_curr=use_curriculum)
  elif args.mode == "test":
      test(model=model,
           env=env,
           num_procs=num_procs,
           num_episodes=args.num_test_episodes)
  else:
      assert False, "Unsupported mode: " + args.mode

  return

if __name__ == '__main__':
  main()
