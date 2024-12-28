# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

abs_path ="/home/avalocal/thesis23/KD/sam2/sam2/configs/sam2.1"

if not GlobalHydra.instance().is_initialized():
    initialize_config_dir(abs_path, version_base="1.2")

    #initialize_config_module("sam2", version_base="1.2")




