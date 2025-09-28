# Blender Rendering Pipeline
This document outlines the process for rendering animations using Blender, specifically for converting SMPL motion data to FBX animations. The pipeline is designed to run on a Windows system and requires the following software:
- Autodesk FBX Converter x64 2013
- Blender 3.4 (Note: 1. Need to install the Auto-Rig Pro plugin. 2. Other Blender versions may not be compatible.)
 
## Step 1: Preprocess .pkl Files
Split a .pkl file containing motion data for multiple dancers into separate files, each containing data for a single dancer.
Run the following command:

    python 1-preProcess_group_smpl.py --input_dir "fk_out"

- `--input_dir`: Directory containing the .pkl file (fk_out)

## Step 2: Convert SMPL Motion Data to FBX Animations
Convert the preprocessed SMPL motion data into FBX animations using a template FBX file `src/ybot.fbx`, which provides skeletal rigging and mesh information. You may replace `src/ybot.fbx` with your preferred template.

Run the following command:

    python 2-ConvertPkl2FBX_SMPL.py --input_dir "fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82" --output_dir "fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out"

## Step 3: Convert FBX Files Using Autodesk FBX Converter
Open Autodesk FBX Converter x64 2013 and convert the output from Step 2 to a compatible FBX version.
![model](Fig/FBX%20Converter.png)
- Default Output Path: `fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out\\FBX 2013\\0.fbx`
## Step 4: Process Dancers and Retarget Animations
Before proceeding, rename the output path from Step 3 to remove the space between FBX and 2013. 
Change: 

    fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out\\FBX 2013\\0.fbx
to:

    fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out\\FBX2013\\0.fbx
Then, automatically convert material colors and retarget animations by running:

    python 4-launch.py --blender_path "PATH\\blender.exe" --main_dir "fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out\\FBX2013" --out_dir "fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\out"

- Replace `PATH\\blender.exe` with the actual path to your Blender executable.
