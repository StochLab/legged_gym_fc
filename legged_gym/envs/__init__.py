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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .stoch3lib.legged_robot_libtraj import LeggedRobotLibTraj
from .stoch3_force_control.legged_robot_libtraj import LeggedRobotLibTrajFC
from .go1.go1_libtraj_fc import Go1LibTrajFC
from .go1_fc.go1_libtraj_fc import Go1LibTraj
from .go1_basic_fc.go1_libtraj_fc import Go1BasicLibTraj
from .go1_basic2.go1_libtraj_fc import Go1BasicLibTraj2
from .go1lib2.legged_robot_libtraj import LeggedRobotGo1LibTraj2
from .go1_fc_new.go1_libtraj_fc import Go1NewLibTraj

from .go1.go1_config import GO1RoughCfg, GO1RoughCfgPPO
from .go1_fc.go1_config import GO1Cfg, GO1CfgPPO
from .go1_basic_fc.go1_config import GO1BasicCfg, GO1BasicCfgPPO
from .go1_basic2.go1_config import GO1BasicCfg2, GO1BasicCfg2PPO
from .go1_fc_new.go1_config import GO1NewCfg, GO1NewCfgPPO
from .stoch3lib.stoch3lib_config import STOCH3LIBRoughCfg, STOCH3LIBRoughCfgPPO
from .stoch3_force_control.stoch3_fc_config import STOCH3ForceControlCfg, STOCH3ForceControlCfgPPO
from .go1lib2.go1lib2_config import GO1LIB2RoughCfg, GO1LIB2RoughCfgPPO

from legged_gym.utils.task_registry import task_registry


task_registry.register("go1", Go1LibTrajFC, GO1RoughCfg(),
                       GO1RoughCfgPPO())
task_registry.register("go1_fc", Go1LibTraj, GO1Cfg(), GO1CfgPPO())
task_registry.register("go1_basic", Go1BasicLibTraj, GO1BasicCfg(), GO1BasicCfgPPO())
task_registry.register("go1_fc_new", Go1NewLibTraj, GO1NewCfg(), GO1NewCfgPPO())
task_registry.register("go1_basic2", Go1BasicLibTraj2, GO1BasicCfg2(), GO1BasicCfg2PPO())
task_registry.register("stoch3lib", LeggedRobotLibTraj, STOCH3LIBRoughCfg(),
                       STOCH3LIBRoughCfgPPO())
task_registry.register("stoch3_fc", LeggedRobotLibTrajFC, STOCH3ForceControlCfg(),
                       STOCH3ForceControlCfgPPO())
task_registry.register("go1lib2", LeggedRobotGo1LibTraj2, GO1LIB2RoughCfg(), GO1LIB2RoughCfgPPO() )
