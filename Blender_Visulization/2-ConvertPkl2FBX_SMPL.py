"""
   Copyright (C) 2017 Autodesk, Inc.
   All rights reserved.

   Use of this software is subject to the terms of the Autodesk license agreement
   provided at the time of installation or download, or which otherwise accompanies
   this software in either electronic or hard copy form.
 
"""

import argparse
import os
from src.FbxReadWriter import FbxReadWrite
from src.SmplObject import SmplObjects

intertype = "Swap" #  Swap Unsimilar Similar
def getArg():
    parser = argparse.ArgumentParser(description='Convert SMPL motion data from PKL files to FBX animations')
    parser.add_argument("--input_dir", type=str,
                    default=f"fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82",
                    help="Directory containing input PKL files with SMPL motion data"
                    )
    parser.add_argument("--fbx_source_path", type=str,
        default="src\\ybot.fbx",
        help="Path to template FBX file containing the skeleton rig"
    )
    parser.add_argument("--output_dir", type=str,
                        default = f"fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out",
                        help="Output directory for generated FBX animation files"
                        )

    return parser.parse_args()


if __name__ == "__main__":
    args = getArg()
    input_dir = args.input_dir
    fbx_source_path = args.fbx_source_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    smplObjects = SmplObjects(input_dir)

    all_entries = os.listdir(output_dir)
    processed_files = [entry for entry in all_entries if os.path.isfile(os.path.join(output_dir, entry))]

    # for pkl_name, smpl_params in tqdm(smplObjects):
    for pkl_name, smpl_params in smplObjects:
        file_name = os.path.basename(pkl_name)
        if file_name[:-4] + ".fbx" in processed_files:
             continue
        fbxReadWrite = FbxReadWrite(fbx_source_path)
        fbxReadWrite.addAnimation(pkl_name, smpl_params)
        pkl_basename = os.path.basename(pkl_name)
        fbxReadWrite.writeFbx(output_dir, pkl_basename) 
        print("done")
