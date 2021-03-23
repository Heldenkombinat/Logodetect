"""
Datasets for different approaches to logo recognition.
All datasets should have the subfolders
    - 'images' - images having logos, all '.jpg'
    - 'annotations' - .txt files, same name as the corresponding image
        Example annotation: (Left, Top, Right, Bottom, Class)
        34 79 100 235 budweiser
        50 123 450 689 cocacola
    - 'exemplars' - clean logo '.jpg' images, not real-world pictures!
        Should overlap as much as possible with the brands present
"""

# Standard library:
import os
import random
from glob import glob
from tabulate import tabulate
from collections import Counter

# Pip packages:
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

# Local:
from .utils import extract_annotation, get_file_info, get_class_name


def cifar10(transform, train):
    return CIFAR10("data", transform=transform, train=train, download=True)


class OneClassDataset(Dataset):
    def __init__(self, transform, root, train_valid, train_prop):
        self.transform = transform
        self.root = os.path.join(os.environ["DATASETS"], root)
        self.train_valid = train_valid
        self.train_prop = train_prop
        self.detections = []
        self.images_paths = self._list_subfolder("images")
        self.annotations_paths = self._list_subfolder("annotations")
        assert len(self.images_paths) == len(self.annotations_paths)

        # Let's go!
        self.preprocess_dataset()

    def _list_subfolder(self, folder):
        data = sorted(glob(os.path.join(self.root, folder, "*")))
        if self.train_valid == "train":
            prop_idx = int(len(data) * self.train_prop)
            return data[:prop_idx]
        if self.train_valid == "valid":
            prop_idx = int(len(data) * (1 - self.train_prop))
            return data[-prop_idx:]

    def preprocess_dataset(self):
        data_used = -1  # -1: All, 1000: Debugging
        for idx, (image, annot) in tqdm(
            enumerate(
                zip(self.images_paths[:data_used], self.annotations_paths[:data_used])
            ),
            desc="Parsing dataset",
            total=len(self.images_paths[:data_used]),
        ):
            # Extract image detections:
            img_detections = self._preprocess_detections(idx, image, annot)
            self.detections.append(img_detections)

    def _preprocess_detections(self, idx, image_path, annotation_path):
        # Strong checks mean we don't mess with the dataset:
        if self._validate_files(image_path, annotation_path):

            classes, boxes = extract_annotation(annotation_path)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # Suppose all instances are not crowd: iscrowd == zeros
            return {
                "boxes": torch.as_tensor(boxes, dtype=torch.int64),
                "area": torch.as_tensor(area, dtype=torch.int64),
                "image_id": torch.as_tensor([idx], dtype=torch.int64),
                "labels": torch.ones((len(classes),), dtype=torch.int64),
                "iscrowd": torch.zeros((len(classes),), dtype=torch.int64),
                # 'classes': classes,
                # 'image_path': image_path,
            }

    def _validate_files(self, image_path, annotation_path):
        # Check if they exist:
        if not os.path.isfile(image_path) or not os.path.isfile(annotation_path):
            raise Exception(
                "One or both of these files is missing: \
                \n{}\n{}".format(
                    image_path, annotation_path
                )
            )
        # Check that they're called the same:
        image_name = get_class_name(image_path)
        annotation_name = get_class_name(annotation_path)
        if image_name != annotation_name:
            raise Exception(
                "The two files don't have the same name: \
                \n{}\n{}".format(
                    image_path, annotation_path
                )
            )
        return True

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        # idx = int(idx)
        image_path = self.images_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.detections[idx]


class StackedDataset(Dataset):
    def __init__(
        self,
        transform,
        root,
        min_area=500,
        min_instances=10,
        percent_same=50,
        exclude=None,
        include=None,
        want_save=True,
    ):
        """
        root:
        transform:
        min_area:
        min_instances:
        percent_same:   Percent_same says how often we combine a detection with the true exemplar
        exclude:        In training, dataset will exclude 'exclude' classes
        include:        In validation, dataset will only include 'include' classes
        mode:
        want_save:      If we use the want_save mode all detections are saved separately
        detections:     Detections will be the good logo detections
        missing:        Missing will be detections for which no exemplar was found
        rejected:       Rejected will be detections whose areas are too small
        rare:           Rare will be detections whose classes don't have enough instances
        class_to_idx:   Class_to_idx will convert from class name to class index - might be handy!
        idx_to_class:
        classes:        Classes will just be a list of the classes
        class_idx:      Class_idx will be a vector holding the class indices
        """
        if include is not None and exclude is not None:
            raise Exception("Options 'exclude' and 'include' are mutually exclusive.")
        self.root = os.path.join(os.environ["DATASETS"], root)
        self.transform = transform
        self.min_area = min_area
        self.min_instances = min_instances
        self.percent_same = percent_same
        self.exclude = exclude
        self.include = include
        if include is None and exclude is None:
            self.mode = None
        else:
            self.mode = "exclude" if exclude else "include"
        self.want_save = want_save
        self.detections = []
        self.missing = []
        self.rejected = []
        self.img_detections = []
        self.img_missing = []
        self.img_rejected = []
        self.rare = []
        self.class_to_idx = []
        self.idx_to_class = []
        self.classes = []
        self.detections_class_idx = []

        # Let's go!
        self.parse_dataset()
        # Print some stats:
        self.dataset_info()

    def parse_dataset(self):
        # Gather all images, annotations, and exemplars:
        images_paths = self._list_subfolder("images")
        annotations_paths = self._list_subfolder("annotations")
        exemplars_paths = self._list_subfolder("exemplars")
        exemplar_classes = [get_class_name(path) for path in exemplars_paths]

        if len(images_paths) != len(annotations_paths):
            raise Exception("Number of images and annotations doesn't match.")

        data_used = -1  # -1: All, 10000: Debugging
        for annotation_path, image_path in tqdm(
            zip(annotations_paths[:data_used], images_paths[:data_used]),
            desc="Parsing dataset",
            total=len(annotations_paths[:data_used]),
        ):
            self._extract_detections(
                image_path, annotation_path, exemplars_paths, exemplar_classes
            )
        # Remove the classes that have too few instances:
        self._prune_dataset()
        # Update the class names and ids:
        self._update_classes()

    def _list_subfolder(self, folder):
        # Include a '.' to ensure the desired format:
        return sorted(glob(os.path.join(self.root, folder, "*.*")))

    def _extract_detections(
        self, image_path, annotation_path, exemplars, exemplar_classes
    ):
        # Strong checks mean we don't mess with the dataset:
        if self._validate_files(image_path, annotation_path):
            # We will also keep track of which brands don't have exemplars:
            self.img_detections, self.img_missing, self.img_rejected = [], [], []
            brands, boxes = extract_annotation(annotation_path)

            # Process each detection:
            for det_idx, (brand, box) in enumerate(zip(brands, boxes)):

                if self._class_allowed(brand):

                    area = (box[3] - box[1]) * (box[2] - box[0])
                    detection = {
                        "class": brand,
                        "image": image_path,
                        "annotation": annotation_path,
                        "name": get_class_name(annotation_path),
                        "area": area,
                    }
                    file_idx = self._find_exemplar_idx(brand, exemplar_classes)

                    # Decide where to save it:
                    self._append_detection(
                        area,
                        detection,
                        file_idx,
                        image_path,
                        brand,
                        box,
                        det_idx,
                        exemplars,
                    )

            # Update global arrays:
            self.detections += self.img_detections
            self.missing += self.img_missing
            self.rejected += self.img_rejected

    def _append_detection(
        self, area, detection, file_idx, image_path, brand, box, det_idx, exemplars
    ):
        if area < self.min_area:
            self.img_rejected.append(detection)

        elif file_idx is None:
            detection["exemplar"] = None
            self.img_missing.append(detection)

        else:
            if self.want_save:
                save_path = self._save_detection(image_path, brand, box, det_idx)
                detection["box"] = None
                detection["image"] = save_path
            else:
                detection["box"] = box
                detection["image"] = image_path
            detection["exemplar"] = exemplars[file_idx]
            self.img_detections.append(detection)

    def _validate_files(self, image_path, annotation_path):
        # Check if they exist:
        if not os.path.isfile(image_path) or not os.path.isfile(annotation_path):
            raise Exception(
                "One or both of these files is missing: \
                \n{}\n{}".format(
                    image_path, annotation_path
                )
            )
        # Check that they're called the same:
        image_name = get_class_name(image_path)
        annotation_name = get_class_name(annotation_path)
        if image_name != annotation_name:
            raise Exception(
                "The two files don't have the same name: \
                \n{}\n{}".format(
                    image_path, annotation_path
                )
            )
        return True

    def _class_allowed(self, brand):
        "Define if this image class must be included."
        if self.mode == "exclude" and brand in self.exclude:
            return False
        elif self.mode == "include" and brand not in self.include:
            return False
        else:
            return True

    def _find_exemplar_idx(self, brand, exemplar_classes):
        try:
            return exemplar_classes.index(brand)
        except ValueError:
            return None

    def _save_detection(self, image_path, brand, box, det_idx):
        _, _, image_name, ext = get_file_info(image_path)
        save_name = "{}_{:03d}{}".format(image_name, det_idx, ext)
        save_folder = os.path.join(self.root, "image_folder", brand)
        save_path = os.path.join(save_folder, save_name)

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        if not os.path.isfile(save_path):
            pil_image = Image.open(image_path).convert("RGB")
            crop = pil_image.crop(box)
            crop.save(save_path)
        return save_path

    def _prune_dataset(self):
        counter = self._count_classes()
        pruned_detections = []
        for detection in self.detections:
            if counter[detection["class"]] < self.min_instances:
                # we add it to the rare detections
                self.rare.append(detection)
            else:
                pruned_detections.append(detection)
        self.detections = pruned_detections

    def _count_classes(self):
        all_classes = [det["class"] for det in self.detections]
        return Counter(all_classes)

    def _update_classes(self):
        self.classes = sorted(set([det["class"] for det in self.detections]))
        self.class_to_idx = {val: idx for idx, val in enumerate(self.classes)}
        self.idx_to_class = {idx: val for idx, val in enumerate(self.classes)}
        # Add the class index in the detection:
        self.detections_class_idx = np.zeros(len(self), dtype=np.int32)
        for idx, detection in enumerate(self.detections):
            self.detections_class_idx[idx] = self.class_to_idx[detection["class"]]
            self.detections[idx]["class_idx"] = self.detections_class_idx[idx]

    def __len__(self, kind="detections"):
        return len(self.__dict__[kind])

    def dataset_info(self):
        headers = [
            "Detections",
            "No exemplars",
            "Too small",
            "Too rare",
            "min_area",
            "min_instances",
        ]
        print("\n[INFO] Dataset information:\n")
        print(
            tabulate(
                [
                    [
                        len(self.detections),
                        len(self.missing),
                        len(self.rejected),
                        len(self.rare),
                        self.min_area,
                        self.min_instances,
                    ]
                ],
                headers,
            )
        )
        print()

    def __getitem__(self, idx, kind="detections", name=None, class_name=None):
        # Get detection:
        detection = self._get_detection(idx, kind, name, class_name)
        # Get the detected logo:
        logo = self._extract_logo(detection)
        # Decide on the exemplar:
        load_same_class = random.randrange(0, 100) <= self.percent_same
        if load_same_class:
            same_class = 1.0
            exemplar = Image.open(detection["exemplar"]).convert("RGB")
        else:
            same_class = 0.0
            # What are the other classes?
            other_classes = [
                name for name in self.classes if name != detection["class"]
            ]
            # Decide on another class to sample from:
            other_class = random.choice(other_classes)
            # Get a detection of that class:
            other_detection = self._get_detection(class_name=other_class)
            exemplar = Image.open(other_detection["exemplar"]).convert("RGB")
        logo = self.transform(logo)
        exemplar = self.transform(exemplar)
        stacked = torch.cat((logo, exemplar))
        return stacked, torch.tensor(same_class).float()

    def _get_detection(self, idx=None, kind="detections", name=None, class_name=None):
        detections = self.__dict__[kind]
        if idx is not None:
            # We return the detection:
            detection = detections[idx]
        elif name is not None:
            # Remove path and extension:
            name = get_class_name(name)
            names = [det["name"] for det in detections]
            try:
                detection = detections[names.index(name)]
            except ValueError:
                raise Exception(
                    "The name {} was not found in the dataset.".format(name)
                )
        elif class_name is not None:
            # Random choice from a given class:
            selection = np.where(
                self.detections_class_idx == self.class_to_idx[class_name]
            )[0]
            class_detections = [detections[idx] for idx in selection]
            detection = random.choice(class_detections)
        else:
            # We return a random one:
            detection = random.choice(detections)
        return detection

    def _extract_logo(self, detection):
        image = Image.open(detection["image"]).convert("RGB")
        if detection["box"] is not None:
            logo = image.crop(detection["box"])
        else:
            logo = image
        return logo
