import os
import torch
import random
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict


ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        dataset_folder: str,
        num_samples: int = 256,
    ):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        number_sample : int, the number of images sampled for descriptor space distribution in a partition.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.augmentation_device = args.augmentation_device
        self.num_samples = num_samples

        # dataset_name should be either "processed", "small" or "raw", if you're using SF-XL
        filename = f"cache/relabel.torch"

        self.scores = torch.load(filename)
        print(self.scores[0])
        self.scores = [s for s in self.scores if len(list(s.values())[0]) > num_samples]

        if self.augmentation_device == "cpu":
            self.transform = T.Compose(
                [
                    T.ColorJitter(
                        brightness=args.brightness,
                        contrast=args.contrast,
                        saturation=args.saturation,
                        hue=args.hue,
                    ),
                    T.RandomResizedCrop(
                        [512, 512],
                        scale=[1 - args.random_resized_crop, 1],
                        antialias=True,
                    ),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Pick num_samples random images from this batch.
        score: dict[str : list[tuple[str, float]]] = self.scores[idx]
        selected_image_paths = random.sample(
            list(score.keys()), self.num_samples
        )  # keys() returns a list of image paths

        score = {k: score[k] for k in selected_image_paths}
        for right_value in score.values():
            # Select the values of the dictionary which key is selected_image_paths
            right_value = [k for k in right_value if k[0] in selected_image_paths]

        # only keep score values as a matrix
        score = torch.tensor([[k[1] for k in v] for v in score.values()])

        image_tensors = []

        for image_path in selected_image_paths:
            image_path = os.path.join(self.dataset_folder, image_path)
            try:
                pil_image = TrainDataset.open_image(image_path)
            except Exception as e:
                logging.info(
                    f"ERROR image {image_path} couldn't be opened, it might be corrupted."
                )
                raise e

            tensor_image = T.functional.to_tensor(pil_image)
            assert tensor_image.shape == torch.Size(
                [3, 512, 512]
            ), f"Image {image_path} should have shape [3, 512, 512] but has {tensor_image.shape}."

            if self.augmentation_device == "cpu":
                tensor_image = self.transform(tensor_image)

            image_tensors.append(
                tensor_image
            )  # Increase the dimension by adding an extra axis

        # Stack all the image tensors along the new axis
        stacked_image_tensors = torch.stack(image_tensors)

        return stacked_image_tensors, score