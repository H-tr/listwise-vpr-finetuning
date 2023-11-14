import os
import cv2
import argparse
from tqdm import tqdm


def split_video_to_frames(source_folder, destination_folder, fps):
    # List all video files in the source folder
    video_files = [f for f in os.listdir(source_folder) if f.endswith(".MOV")]

    for video_file in video_files:
        # Create a subfolder for each video in the destination folder
        video_name = os.path.splitext(video_file)[0]
        video_destination_folder = os.path.join(destination_folder, video_name)
        os.makedirs(video_destination_folder, exist_ok=True)

        # Path to the input video
        video_path = os.path.join(source_folder, video_file)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set the desired frames per second (fps)
        new_fps = int(fps)

        # Calculate the interval to capture frames
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / new_fps)

        # Loop through the frames and save them with a progress bar
        frame_count = 0
        with tqdm(
            total=total_frames // frame_interval,
            desc=f"Processing {video_file}",
            unit="frames",
        ) as pbar:
            while frame_count < total_frames // frame_interval:
                # Capture the frame
                ret, frame = cap.read()

                if not ret:
                    break

                # Save frame in the destination folder
                frame_destination_path = os.path.join(
                    video_destination_folder, f"{video_name}_frame_{frame_count}.jpg"
                )
                cv2.imwrite(frame_destination_path, frame)

                # Move to the next frame
                frame_count += 1

                # Skip frames based on the interval
                for _ in range(frame_interval - 1):
                    cap.read()

                # Update the progress bar
                pbar.update(1)

        # Release the video capture object
        cap.release()

        print(
            f"Frames extracted from {video_file} and saved to {video_destination_folder}"
        )


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

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the function to split videos into frames
    split_video_to_frames(args.source, args.destination, args.fps)
