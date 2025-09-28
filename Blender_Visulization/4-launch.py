import os
import subprocess
import glob
import argparse
from tqdm import tqdm

def process_files(blender_path, python_script, smpl_dir, main_dir, out_dir):
    """
    Process all FBX files in the input directory using Blender
    
    Args:
        blender_path (str): Path to Blender executable
        python_script (str): Path to Python script to execute
        smpl_dir (str): Path to SMPL template FBX
        main_dir (str): Directory with input FBX files
        out_dir (str): Output directory for processed files
    """
    os.makedirs(out_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(main_dir, "*.fbx"))
    
    if not files:
        print(f"No FBX files found in directory: {main_dir}")
        return
    
    for count, file_path in enumerate(tqdm(sorted(files), desc="Processing FBX files")):
        cmd = [
            blender_path, 
            "-b",
            "-P",
            python_script,
            "--",
            smpl_dir,
            file_path,
            out_dir,
            main_dir,
            str(count)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing file {file_path}: {e}")
        except FileNotFoundError:
            print(f"Blender not found at {blender_path}. Please check the path.")
            return

def main():
    """Parse command line arguments and start processing"""
    parser = argparse.ArgumentParser(
        description="Process FBX files using Blender and a Python script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--blender_path", type=str, default='PATH\\blender.exe',
                        help="Path to Blender executable")
    parser.add_argument("--python_script", type=str, default='smpl2smplforLaunch.py',
                        help="Path to Python script for Blender")
    parser.add_argument("--smpl_dir", type=str, default='src\\ybot.fbx',
                        help="Path to SMPL template FBX file")
    parser.add_argument("--main_dir", type=str, default='fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\fbx_out\\FBX2013',
                        help="Directory with input FBX files")
    parser.add_argument("--out_dir", type=str, default='fk_out\\1_17_tl5KkCiOGFw_02_0_1920_slice82\\out',
                        help="Output directory for processed files")
    
    args = parser.parse_args()
    

    for path_arg in [args.blender_path, args.python_script, args.smpl_dir]:
        if not os.path.exists(path_arg):
            print(f"Error: Path does not exist: {path_arg}")
            return
    
    if not os.path.exists(args.main_dir):
        print(f"Input directory not found: {args.main_dir}")
        return
    
    # Start processing
    process_files(
        blender_path=args.blender_path,
        python_script=args.python_script,
        smpl_dir=args.smpl_dir,
        main_dir=args.main_dir,
        out_dir=args.out_dir
    )
    
    print("All files processed successfully!")

if __name__ == "__main__":
    main()