import glob
import os
import pickle
from typing import Dict, Tuple

import numpy as np


class SmplObjects(object):
    joints = [
        "m_avg_Pelvis",
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ]

    # # mixamo1
    # joints = [
    #     "mixamorig1:Hips",
    #     "mixamorig1:LeftUpLeg",
    #     "mixamorig1:RightUpLeg",
    #     "mixamorig1:Spine",
    #     "mixamorig1:LeftLeg",
    #     "mixamorig1:RightLeg",
    #     "mixamorig1:Spine1",
    #     "mixamorig1:LeftFoot",
    #     "mixamorig1:RightFoot",
    #     "mixamorig1:Spine2",
    #     "mixamorig1:LeftToeBase",
    #     "mixamorig1:RightToeBase",
    #     "mixamorig1:Neck",
    #     # "m_avg_L_Collar",
    #     # "m_avg_R_Collar",
    #     "mixamorig1:Head",
    #     "mixamorig1:LeftShoulder",
    #     "mixamorig1:RightShoulder",
    #     "mixamorig1:LeftArm",
    #     "mixamorig1:RightArm",
    #     "mixamorig1:LeftForeArm",
    #     "mixamorig1:RightForeArm",
    #     "mixamorig1:LeftHand",
    #     "mixamorig1:RightHand",
    # ]

    # # mixamo
    # joints = [
    #     "mixamorig:Hips",
    #     "mixamorig:LeftUpLeg",
    #     "mixamorig:RightUpLeg",
    #     "mixamorig:Spine",
    #     "mixamorig:LeftLeg",
    #     "mixamorig:RightLeg",
    #     "mixamorig:Spine1",
    #     "mixamorig:LeftFoot",
    #     "mixamorig:RightFoot",
    #     "mixamorig:Spine2",
    #     "mixamorig:LeftToeBase",
    #     "mixamorig:RightToeBase",
    #     "mixamorig:Neck",
    #     "m_avg_L_Collar",
    #     "m_avg_R_Collar",
    #     "mixamorig:Head",
    #     "mixamorig:LeftShoulder",
    #     "mixamorig:RightShoulder",
    #     "mixamorig:LeftArm",
    #     "mixamorig:RightArm",
    #     "mixamorig:LeftForeArm",
    #     "mixamorig:RightForeArm",
    #     "mixamorig:LeftHand",
    #     "mixamorig:RightHand",
    # ]


    def __init__(self, read_path):
        self.files = {}

        paths = sorted(glob.glob(os.path.join(read_path, "*.pkl")))
        for path in paths:
            filename = path.split("/")[-1]
            with open(path, "rb") as fp:
                data = pickle.load(fp)
            self.files[filename] = {
                "smpl_poses": data["smpl_poses"],
                "smpl_trans": data["smpl_trans"],
            }
        self.keys = [key for key in self.files.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        key = self.keys[idx]
        return key, self.files[key]
