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

from .go1_wrench1.go1_libtraj_fc import Go1Wrench1LibTraj
from .go1_wrench2.go1_libtraj_fc import Go1Wrench2LibTraj
from .go1_wrench3.go1_libtraj_fc import Go1Wrench3LibTraj
from .go1_wrench4.go1_libtraj_fc import Go1Wrench4LibTraj
from .go1_wrench5.go1_libtraj_fc import Go1Wrench5LibTraj
from .go1_wrench6.go1_libtraj_fc import Go1Wrench6LibTraj
from .go1_wrench7.go1_libtraj_fc import Go1Wrench7LibTraj
from .go1_wrench8.go1_libtraj_fc import Go1Wrench8LibTraj
from .go1_wrench9.go1_libtraj_fc import Go1Wrench9LibTraj

from .go1_grf1.go1_libtraj_fc import Go1grf1LibTraj

from .go1_wrench1.go1_config import GO1Wrench1Cfg, GO1Wrench1CfgPPO
from .go1_wrench2.go1_config import GO1Wrench2Cfg, GO1Wrench2CfgPPO
from .go1_wrench3.go1_config import GO1Wrench3Cfg, GO1Wrench3CfgPPO
from .go1_wrench4.go1_config import GO1Wrench4Cfg, GO1Wrench4CfgPPO
from .go1_wrench5.go1_config import GO1Wrench5Cfg, GO1Wrench5CfgPPO
from .go1_wrench6.go1_config import GO1Wrench6Cfg, GO1Wrench6CfgPPO
from .go1_wrench7.go1_config import GO1Wrench7Cfg, GO1Wrench7CfgPPO
from .go1_wrench8.go1_config import GO1Wrench8Cfg, GO1Wrench8CfgPPO
from .go1_wrench9.go1_config import GO1Wrench9Cfg, GO1Wrench9CfgPPO

from .go1_grf1.go1_config import GO1grf1Cfg, GO1grf1CfgPPO

from .go1_fc_new.go1_config import GO1NewCfg, GO1NewCfgPPO
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.task_registry_bc import task_registry_bc

task_registry.register("go1_wrench1", Go1Wrench1LibTraj, GO1Wrench1Cfg(), GO1Wrench1CfgPPO())
task_registry.register("go1_wrench2", Go1Wrench2LibTraj, GO1Wrench2Cfg(), GO1Wrench2CfgPPO())
task_registry.register("go1_wrench3", Go1Wrench3LibTraj, GO1Wrench3Cfg(), GO1Wrench3CfgPPO())
task_registry.register("go1_wrench4", Go1Wrench4LibTraj, GO1Wrench4Cfg(), GO1Wrench4CfgPPO())
task_registry.register("go1_wrench5", Go1Wrench5LibTraj, GO1Wrench5Cfg(), GO1Wrench5CfgPPO())
task_registry.register("go1_wrench6", Go1Wrench6LibTraj, GO1Wrench6Cfg(), GO1Wrench6CfgPPO())
task_registry.register("go1_wrench7", Go1Wrench7LibTraj, GO1Wrench7Cfg(), GO1Wrench7CfgPPO())
task_registry.register("go1_wrench8", Go1Wrench8LibTraj, GO1Wrench8Cfg(), GO1Wrench8CfgPPO())
task_registry.register("go1_wrench9", Go1Wrench9LibTraj, GO1Wrench9Cfg(), GO1Wrench9CfgPPO())

task_registry.register("go1_grf1", Go1grf1LibTraj, GO1grf1Cfg(), GO1grf1CfgPPO())
