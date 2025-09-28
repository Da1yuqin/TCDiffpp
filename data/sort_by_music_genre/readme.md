# Music Dataset Preprocessing

This folder contains two Python scripts designed to organize music dataset files by genre and extract sequence IDs for later use.

> 📌 Note: The `test` and `train` folders should be processed in the same way as the `val` folder (first run `split_music_genres.py`, then run `extract_ids.py`).


# 1. Scripts Overview

```bash
python 1_split_music_genres.py
```

- Purpose:

Splits the raw dataset CSV file (e.g., ``val_labels.csv``) into multiple CSV files, each containing samples from a specific target music genre (Pop, Electronic, Indian, Latin, R&B, Reggae, Rap).

- Output:

A folder of genre-specific CSV files, e.g.:

```bash
./val/Pop.csv
./val/Electronic.csv
./val/Indian.csv
```

extract_ids.py

- Purpose:

Extracts the first column (ID field) from a given genre CSV file and saves it into a .txt file, one ID per line.

- Output:

A text file containing sequence IDs, e.g.:
```bash
./val/Electronic_split_sequence_names.txt
```

# 2. Output Format
```bash
python 2_csv2txt.py
```


After running both scripts, the data will be organized as follows:
```bash
/val
 ├── Pop.csv
 ├── Electronic.csv
 ├── Indian.csv
 ├── Latin.csv
 ├── R&B.csv
 ├── Reggae.csv
 ├── Rap.csv
 ├── Electronic_split_sequence_names.txt
 └── ...
```

✨ With this workflow, you first split the dataset by genre, then extract IDs for further experiments or data loading.