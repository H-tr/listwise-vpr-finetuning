import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_test_dataset")
import random
import shutil
import logging

def generate_test_dataset(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Create a new folder with the name of the original folder + "_test"
    new_folder_path = folder_path + "_test"
    # if exists, delete the folder
    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    # create the folder
    os.mkdir(new_folder_path)

    # Create query and database if they don't exist
    if not os.path.exists(os.path.join(new_folder_path, "query")):
        os.mkdir(os.path.join(new_folder_path, "query"))
        os.mkdir(os.path.join(new_folder_path, "database"))

    # Copy the first 20 images from the shuffled list to the new folder as query images
    for i in range(20):
        image_file = image_files[i]
        shutil.copy(os.path.join(folder_path, image_file), os.path.join(new_folder_path, "query", image_file))

    # Copy the rest of the images to the new folder as database images
    for i in range(20, len(image_files)):
        image_file = image_files[i]
        shutil.copy(os.path.join(folder_path, image_file), os.path.join(new_folder_path, "database", image_file))

    # Save the new folder in the same directory as the original folder
    os.rename(new_folder_path, folder_path + "_test")

def main():
    root_folder = "data/processed"
    
    # remove all the folder ending with "_test"
    for folder in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, folder)) and folder.endswith("_test"):
            shutil.rmtree(os.path.join(root_folder, folder))
            
    for folder in os.listdir(root_folder):
        logger.info(f"Generating test dataset for {folder}")
        if os.path.isdir(os.path.join(root_folder, folder)):
            generate_test_dataset(os.path.join(root_folder, folder))

if __name__ == "__main__":
    main()
