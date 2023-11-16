import os
import logging
from glob import glob
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_images_paths(dataset_folder, get_abs_path=False):
    """
    Find images within 'dataset_folder' and return their relative paths as a list.
    If there is a file 'dataset_folder'_images_paths.txt, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders can be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    get_abs_path : bool, if True return absolute paths, otherwise remove
        dataset_folder from each path

    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    # traverse the folder and save all the paths of images in a list
    images_paths = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".jpg"):
                images_paths.append(os.path.join(root, file))

    if not get_abs_path:  # Remove dataset_folder from the path
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]

    return images_paths
