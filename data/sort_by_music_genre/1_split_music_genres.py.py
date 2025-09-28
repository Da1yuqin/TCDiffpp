import os
from collections import defaultdict

def split_music_genres(input_path, output_dir):
    """
    Split music data according to music_genre
    
    Parameters:
    input_path (str): Input file path
    output_dir (str): Output directory path
    """
    # Define target music genres
    target_genres = {'Pop', 'Electronic', 'Indian', 'Latin', 'R&B', 'Reggae', 'Rap'}
    
    # Check if the input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    
    # Create the output directory (if it does not exist)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a dictionary to store data for different genres
    genre_data = defaultdict(list)
    header = ""

    # Read the original file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # Read the header line
            header = f.readline().strip()
            
            # Process data lines
            for line_number, line in enumerate(f, 2):  # start from line 2
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 3:
                    print(f"Warning: Line {line_number} has invalid format - '{line}'")
                    continue
                    
                genre = parts[1].strip()
                if genre in target_genres:
                    genre_data[genre].append(line)
    
    except Exception as e:
        print(f"Error while processing file: {str(e)}")
        return

    # Create a separate file for each genre
    file_count = 0
    for genre, lines in genre_data.items():
        # Clean file name (remove special characters)
        clean_genre = genre.replace('&', '').replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{clean_genre}.csv")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header + '\n')
                f.write('\n'.join(lines))
            file_count += 1
            print(f"Created: {output_path} ({len(lines)} rows of data)")
        except Exception as e:
            print(f"Failed to create file {output_path}: {str(e)}")

    # Create empty files for genres with no data
    for genre in target_genres - set(genre_data.keys()):
        clean_genre = genre.replace('&', '').replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{clean_genre}.csv")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header + '\n')
            file_count += 1
            print(f"Created empty file: {output_path}")
        except Exception as e:
            print(f"Failed to create empty file {output_path}: {str(e)}")
    
    print(f"\nProcessing completed! A total of {file_count} files have been created in directory: {output_dir}")

if __name__ == "__main__":
    # Configure input and output paths (modify to your actual paths)
    input_file = "./data/AIOZ-dataset/val_labels.csv"  # Full path of the input file
    output_directory = "./data/AIOZ-dataset/sort_by_music_genre/val"  # Output directory path
    
    # Execute classification
    split_music_genres(input_file, output_directory)
