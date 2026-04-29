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
from .base.legged_robot_wtw import LeggedRobotwtw
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1_stairs_config import A1StairsCfg, A1StairsCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO

from .aliengo.aliengo_config import AlienGoRoughCfg, AlienGoRoughCfgPPO
from .aliengo.aliengo_stairs_config import AlienGoStairsCfg, AlienGoStairsCfgPPO
from .aliengo.aliengo_stairs_amp_config import AlienGoStairsAmpCfg, AlienGoStairsAmpCfgPPO
from .aliengo.aliengo_recover_config import AlienGoRoughRecoverCfg, AlienGoRoughRecoverCfgPPO
from .aliengo.aliengo_lidar_config import AlienGoFlatLidarCfg, AlienGoFlatLidarCfgPPO

from .lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO
from .lite3.lite3_stairs_config import Lite3StairsCfg, Lite3StairsCfgPPO
from .lite3.lite3_stairs_amp_config import Lite3StairsAmpCfg, Lite3StairsAmpCfgPPO
from .lite3.lite3_recover_config import Lite3RoughRecoverCfg, Lite3RoughRecoverCfgPPO
from .lite3.lite3_lidar_config import Lite3FlatLidarCfg, Lite3FlatLidarCfgPPO
from .lite3.lite3_jump_config import Lite3JumpCfg, Lite3JumpCfgPPO


from .cc1.cc1_config import Cc1RoughCfg, Cc1RoughCfgPPO
from .cc1.cc1_wtw_config import Cc1RoughwtwCfg, Cc1RoughwtwCfgPPO
from .cc1.cc1_jump_config import Cc1JumpCfg, Cc1JumpCfgPPO 
from .cc1.cc1_stairs_wtw_config import Cc1StairswtwCfg, Cc1StairswtwCfgPPO 
from .cc1.cc1_jump_wtw_config import Cc1JumpwtwCfg, Cc1JumpwtwCfgPPO  
from .cc1.cc1_jump_high_wtw_config import Cc1JumpHighwtwCfg, Cc1JumpHighwtwCfgPPO
from .cc1.cc1_jump1_config import Cc1Jump1Cfg, Cc1Jump1CfgPPO  
from .cc1.cc1_stairs_config import Cc1StairsCfg, Cc1StairsCfgPPO
from .cc1.cc1_recover_config import Cc1RoughRecoverCfg, Cc1RoughRecoverCfgPPO
from .cc1.cc1_velocity_config import Cc1VelocityCfg, Cc1VelocityCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_stairs", LeggedRobot, A1StairsCfg(), A1StairsCfgPPO() )
task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "aliengo", LeggedRobot, AlienGoRoughCfg(), AlienGoRoughCfgPPO() )
task_registry.register( "aliengo_stairs", LeggedRobot, AlienGoStairsCfg(), AlienGoStairsCfgPPO() )
task_registry.register( "aliengo_stairs_amp", LeggedRobot, AlienGoStairsAmpCfg(), AlienGoStairsAmpCfgPPO() )
task_registry.register( "aliengo_recover", LeggedRobot, AlienGoRoughRecoverCfg(), AlienGoRoughRecoverCfgPPO() )
task_registry.register("aliengo_lidar", LeggedRobot, AlienGoFlatLidarCfg(), AlienGoFlatLidarCfgPPO())

task_registry.register( "lite3", LeggedRobot, Lite3RoughCfg(), Lite3RoughCfgPPO() )
task_registry.register( "lite3_stairs", LeggedRobot, Lite3StairsCfg(), Lite3StairsCfgPPO() )
task_registry.register( "lite3_stairs_amp", LeggedRobot, Lite3StairsAmpCfg(), Lite3StairsAmpCfgPPO() )
task_registry.register( "lite3_recover", LeggedRobot, Lite3RoughRecoverCfg(), Lite3RoughRecoverCfgPPO() )
task_registry.register( "lite3_lidar", LeggedRobot, Lite3FlatLidarCfg(), Lite3FlatLidarCfgPPO() )
task_registry.register( "lite3_jump", LeggedRobot, Lite3JumpCfg(), Lite3JumpCfgPPO() )


task_registry.register( "cc1", LeggedRobot, Cc1RoughCfg(), Cc1RoughCfgPPO() )
task_registry.register( "cc1_wtw", LeggedRobotwtw, Cc1RoughwtwCfg(), Cc1RoughwtwCfgPPO() )
task_registry.register( "cc1_jump", LeggedRobot, Cc1JumpCfg(), Cc1JumpCfgPPO() )
task_registry.register( "cc1_jump_wtw", LeggedRobotwtw, Cc1JumpwtwCfg(), Cc1JumpwtwCfgPPO() )
task_registry.register( "cc1_jump_high_wtw", LeggedRobotwtw, Cc1JumpHighwtwCfg(), Cc1JumpHighwtwCfgPPO() )
task_registry.register( "cc1_jump1", LeggedRobot, Cc1Jump1Cfg(), Cc1Jump1CfgPPO() )
task_registry.register( "cc1_stairs", LeggedRobot, Cc1StairsCfg(), Cc1StairsCfgPPO() )
task_registry.register( "cc1_stairs_wtw", LeggedRobotwtw, Cc1StairswtwCfg(), Cc1StairswtwCfgPPO() )
task_registry.register( "cc1_recover", LeggedRobot, Cc1RoughRecoverCfg(), Cc1RoughRecoverCfgPPO() )
task_registry.register( "cc1_velocity", LeggedRobot, Cc1VelocityCfg(), Cc1VelocityCfgPPO() )
