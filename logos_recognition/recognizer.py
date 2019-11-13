"Classes for detecting and recognizing logos."

# Standard library:
import os

# Pip packages:
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
import skvideo.io
from PIL import Image

# Github repos:

# Logos-Recognition:
from logos_recognition.detector import Detector
from logos_recognition.classifier import Classifier
from logos_recognition.utils import (get_class_name,
                                     open_resize_and_load_gpu)
from logos_recognition.constants import QUERY_LOGOS


class Recognizer(object):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.detector = Detector()
        self.classifier = Classifier()
        self.video = None
        self.frame_duration = None
        self.total_frames = None
        self.video_area = None
        self.query_logos = None
        self.query_logos_names = None

    def recognize(self, load_name, output_path, query_logos):
        "Add documentation."
        # load query logos
        self.load_query_logos(query_logos)
        # deal with the video
        self.set_video_source(load_name)
        # Get a handle for the entire video:
        subclip_handle = self.video.subclip(0, self.video.end)

        # Extract detections frame by frame:
        recognitions = []
        for frame in tqdm(subclip_handle.iter_frames(),
                          total=self.total_frames,
                          desc="Processing video"):
            # Detect all classes:
            detections = self.detector.predict(frame)
            # Select the desired classes:
            classifications = self.classifier.predict(
                detections, frame, self.query_logos)
            recognitions.append(classifications)

        # Apply filters/enhancements to recognitions:
        recognitions = self.process_recognitions(recognitions)
        # Draw the final detections:
        self.draw_and_save_video(recognitions, output_path)

    def load_query_logos(self, query_logos):
        "Add documentation."
        # save the class names
        query_logos_names = [get_class_name(path)
                             for path in query_logos]
        query_logos = [open_resize_and_load_gpu(path)
                       for path in query_logos]

        self.query_logos_names = query_logos_names
        self.query_logos = query_logos

    def set_video_source(self, load_name):
        "Add documentation."
        self.video = VideoFileClip(load_name)
        self.frame_duration = (1 / self.video.fps)
        self.total_frames = int(self.video.reader.nframes)
        self.video_area = self.video.size[0] * self.video.size[1]

    def process_recognitions(self, recognitions):
        "Add documentation."
        return recognitions

    def draw_and_save_video(self, recognitions, output_path):
        "Add documentation."
        subclip_handle = self.video.subclip(0, self.video.end)
        writer = skvideo.io.FFmpegWriter(
            output_path, outputdict={"-pix_fmt": 'yuv420p'})
        for idx, frame in enumerate(tqdm(subclip_handle.iter_frames(),
                                         total=self.total_frames,
                                         desc="Rendering video")):
            result = self.overlay_boxes(frame, recognitions[idx])
            img = np.array(result)
            writer.writeFrame(img)
        writer.close()

    def overlay_boxes(self, image, recognitions):
        "Add documentation."
        boxes = recognitions["boxes"]
        scores = recognitions["scores"]
        labels = recognitions["labels"]

        # determine class colors
        cmap = plt.cm.get_cmap("jet", len(self.query_logos_names))

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            color = tuple(np.array(cmap(label))[:3] * 255)
            top_left, bottom_right = box[:2], box[2:]
            image = cv2.rectangle(image, tuple(top_left),
                                  tuple(bottom_right), color, 2)
            text = template.format(self.query_logos_names[label], score)
            cv2.putText(image, text, tuple(top_left),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return image
