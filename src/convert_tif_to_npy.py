import os
import glob
import numpy as np
import tifffile as tiff

def convert_tif_to_npy(directory, suffix='.tif'):
    """
    Converts all TIFF mask files in the given directory to .npy format.
    Output .npy files will have the same basename as the TIFF files.
    """
    tif_paths = glob.glob(os.path.join(directory, f'*{suffix}'))
    for tif_path in tif_paths:
        mask = tiff.imread(tif_path)
        base = os.path.splitext(os.path.basename(tif_path))[0]
        npy_path = os.path.join(directory, f'{base}.npy')
        np.save(npy_path, mask)
        print(f'Converted {tif_path} -> {npy_path}')

if __name__ == "__main__":
    # TODO: update this to your masks folder
    data_dir = "/Users/lucas/Documents/GitHub/ECT_template_1/data/processed/train_cp"
    convert_tif_to_npy(data_dir)
