import argparse
import pickle
import os
import glob

def split_pkl_file(file_path, output_file_prefix, name):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    dancer_num, seq_len, _, _ = data['full_pose'].shape

    split_data = data
    split_data['smpl_poses'] = split_data['smpl_poses'].reshape(seq_len, dancer_num,-1)
    split_data['smpl_trans'] = split_data['smpl_trans'].reshape(seq_len, dancer_num,-1)

    for i in range(dancer_num):
        sub_data = {}
        sub_data['smpl_poses'] = split_data['smpl_poses'][:,i,:]
        sub_data['smpl_trans'] = split_data['smpl_trans'][:,i,:]
        sub_data['full_pose'] = split_data['full_pose'][i:i+1,:,:]
        
        output_file_dir = os.path.join(output_file_prefix, os.path.splitext(name)[0])
        os.makedirs(output_file_dir, exist_ok=True)
        output_file_path = os.path.join(output_file_dir, f"{i}.pkl")
        with open(output_file_path, 'wb') as file:
            pickle.dump(sub_data, file)

def main(input_dir):
    """
    Process all PKL files in the specified directory
    
    Args:
        input_dir (str): Directory containing PKL files to process
    """
    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"No PKL files found in directory: {input_dir}")
        return

    for file_path in pkl_files:
        file_name = os.path.basename(file_path)
        split_pkl_file(file_path, input_dir, file_name)

    print("File splitting completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split multi-dancer PKL files into individual dancer files')
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='fk_out',
        help='Directory containing PKL files to process'
    )
    
    args = parser.parse_args()
    
    main(args.input_dir)