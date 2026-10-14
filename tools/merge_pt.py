import torch
from pathlib import Path
from tqdm import tqdm


def merge_pt_files(folder_path, output_filename):
    merged_dict = {}
    folder = Path(folder_path)

    file_list = list(folder.glob("*.pt"))

    print(f"find {len(file_list)} files，start to extracting...")

    for file_path in tqdm(file_list):
        # load preprocess results
        data = torch.load(file_path, map_location='cpu')
        file_id = file_path.stem

        merged_dict[file_id] = data

    print(f"saving merged file to : {output_filename}")
    torch.save(merged_dict, output_filename)
    print("task completed！")

# run
merge_pt_files('./output/relation_baseline/preprocess', './output/relation_baseline/merged_data-long-fuzzy-refine.pt')
