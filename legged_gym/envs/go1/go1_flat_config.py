# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go1.go1_rough_config import Go1RoughCfg, Go1RoughCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Go1FlatCfg( Go1RoughCfg ):
    train_jump = True
    draw_goal = False

    class env( Go1RoughCfg.env ):
        num_observations = 48
        

    class terrain( Go1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False


    class asset( Go1RoughCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new.urdf'
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( Go1RoughCfg.rewards ):


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.

        max_contact_force = 100. # forces above this value are penalized
        
        
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
      
        only_positive_rewards = False
        class scales:
            ang_vel_xy = -0.3 #-0.05
            
            dof_acc = -2.5e-7
            action_rate = -0.001
            collision = -0.5
            orientation = -2.
            # termination = 50.
            
            # new reward funcs to be formulated
            
            height_off_ground = 0.
            xy_proximity = 0.
            # tracking_yaw = 1.5
            # tracking_goal_vel = 1.5
            # ang_vel_z = -2.5e-5 #-0.05
            
            # zeros ones rewards are disabled
            lin_vel_z = 0.0 
            torques = -0.0

            
            feet_stumble = -0.0 
            # base_height = -0.001 
            stand_still = -0.
            
            dof_vel = -0.
            
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            ang_vel_xy = -0.05
            feet_air_time =  .05

    class commands(LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 8 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error) # added jump_satrt_z_vel
        resampling_time = 4. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            jump_start_z_vel = [0.5, 1.2] # [m/s]
            # just height
            jump_start_z = [.6, 0.9] # [m]
            # also adding relative x,y position w.r.t init dog position 
            jump_start_x = [0.5,1.] # relative coordinate of the static point
            jump_start_y = [-0.4, 0.4] # relative cooridnate fo the static point


class Go1FlatCfgPPO( Go1RoughCfgPPO ):
    class policy( Go1RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( Go1RoughCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( Go1RoughCfgPPO.runner ):
        run_name = ''
        load_run = -1
        experiment_name = 'flat_go1'

  
