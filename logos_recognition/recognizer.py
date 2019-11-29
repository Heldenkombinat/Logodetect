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
# from logos_recognition.classifier import Classifier
from logos_recognition.classifier import KNNClassifier
from logos_recognition.utils import (get_class_name,
                                     open_resize_and_load_gpu)
from logos_recognition.constants import QUERY_LOGOS


class Recognizer(object):
    "Add documentation."

    def __init__(self):
        "Add documentation."
        self.detector = Detector()
        # self.classifier = Classifier()
        self.classifier = KNNClassifier()
        self.video = None
        self.frame_duration = None
        self.total_frames = None
        self.video_area = None
        self.query_logos = None
        self.query_logos_names = None

    def recognize(self, load_name, output_path, query_logos):
        '''
        classifications = {
            'boxes': [] or 2D array (float32),
            'labels': [] or 1D array (int),
            'scores': [] or 1D array (float32),
            'brands': [] or 1D array (str)
            }
        '''
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
            if len(detections['boxes']) == 0:
                detections['brands'] = []
                recognitions.append(detections)
                continue
            # Select the desired classes:
            classifications = self.classifier.predict(
                detections, frame, self.query_logos)
            recognitions.append(classifications)

        # Draw the final detections:
        self.draw_and_save_video(recognitions, output_path)

    def load_query_logos(self, query_logos):
        "Add documentation."
        # save the class names
        self.query_logos_names = [get_class_name(path)
                                  for path in query_logos]
        self.query_logos = [open_resize_and_load_gpu(path)
                            for path in query_logos]

    def set_video_source(self, load_name):
        "Add documentation."
        self.video = VideoFileClip(load_name)
        self.frame_duration = (1 / self.video.fps)
        self.total_frames = int(self.video.reader.nframes)
        self.video_area = self.video.size[0] * self.video.size[1]

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
        boxes = recognitions['boxes']
        labels = recognitions['labels']
        scores = recognitions['scores']
        brands = recognitions['brands']

        # Define class colors:
        # cmap = plt.cm.get_cmap("jet", len(self.query_logos_names))
        cmap = plt.cm.get_cmap("jet", len(brands))
        font = cv2.FONT_HERSHEY_SIMPLEX

        template = "{}: {:.2f}"
        for box, label, score, brand in zip(boxes, labels, scores, brands):

            top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
            color = tuple(np.array(cmap(label))[:3] * 255)
            image = cv2.rectangle(image, top_left, bottom_right, color, 2)
            # text = template.format(self.query_logos_names[label], score)
            text = template.format(brand, score)
            cv2.putText(image, text, top_left, font, 0.7, color, 2)
            
        return image
