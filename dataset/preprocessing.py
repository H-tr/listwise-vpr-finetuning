import os
import cv2
import argparse
import logging
from rich.progress import Progress
import shutil
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process dataset")


def split_video_to_frames(source_folder, destination_folder, fps):
    # List all video files in the source folder
    video_files = [f for f in os.listdir(source_folder) if f.endswith(".MOV")]
    os.makedirs(destination_folder, exist_ok=True)
    logger.info(f"destination folder {destination_folder} created!")

    for video_file in video_files:
        # Create a subfolder for each video in the destination folder
        video_name = os.path.splitext(video_file)[0]
        video_destination_folder = os.path.join(destination_folder, video_name)
        # if exists, delete the folder
        if os.path.exists(video_destination_folder):
            shutil.rmtree(video_destination_folder)
        os.makedirs(video_destination_folder, exist_ok=True)

        # Path to the input video
        video_path = os.path.join(source_folder, video_file)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the desired frames per second (fps)
        new_fps = int(fps)

        # Calculate the interval to capture frames
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / new_fps)

        # Loop through the frames and save them with a progress bar
        frame_count = 0
        with Progress() as progress:
            task1 = progress.add_task(
                f"[red]Processing {video_file}", total=total_frames // frame_interval
            )
            while frame_count < total_frames // frame_interval:
                # Capture the frame
                ret, frame = cap.read()

                if not ret:
                    break

                # Save frame in the destination folder
                num_frame = str(frame_count).zfill(5)
                frame_destination_path = os.path.join(
                    video_destination_folder,
                    f"{video_name}_frame_" + num_frame + ".jpg",
                )
                cv2.imwrite(frame_destination_path, frame)

                # Move to the next frame
                frame_count += 1

                # Skip frames based on the interval
                for _ in range(frame_interval - 1):
                    cap.read()

                progress.update(task1, advance=1)

        # Release the video capture object
        cap.release()


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
        shutil.copy(
            os.path.join(folder_path, image_file),
            os.path.join(new_folder_path, "query", image_file),
        )

    # Copy the rest of the images to the new folder as database images
    for i in range(20, len(image_files)):
        image_file = image_files[i]
        shutil.copy(
            os.path.join(folder_path, image_file),
            os.path.join(new_folder_path, "database", image_file),
        )

    # Save the new folder in the same directory as the original folder
    os.rename(new_folder_path, folder_path + "_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a video into frames and save them in separate folders."
    )
    parser.add_argument(
        "--source", required=True, help="Path to the source folder containing videos."
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Path to the destination folder for saving frames.",
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Desired frames per second (fps)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate test dataset for the folder",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the function to split videos into frames
    split_video_to_frames(args.source, args.destination, args.fps)

    # remove all the folder ending with "_test"
    for folder in os.listdir(args.destination):
        if os.path.isdir(os.path.join(args.destination, folder)) and folder.endswith(
            "_test"
        ):
            shutil.rmtree(os.path.join(args.destination, folder))

    for folder in os.listdir(args.destination):
        logger.info(f"Generating test dataset for {folder}")
        if os.path.isdir(os.path.join(args.destination, folder)):
            generate_test_dataset(os.path.join(args.destination, folder))
