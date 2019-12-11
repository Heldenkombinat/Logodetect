"Classes for detecting and recognizing logos."

# Standard library:
import os
import glob
from importlib import import_module

# Pip packages:
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip, ImageClip, concatenate
from moviepy.audio.fx.volumex import volumex
from PIL import Image

# Current library:
from logos_recognition.utils import get_class_name, open_resize_and_load_gpu
from logos_recognition.constants import (DETECTOR, CLASSIFIER, DETECTOR_DEVICE,
                                         USE_CLASSIFIER, BRAND_LOGOS, IMAGE_RESIZE)



class Recognizer(object):
    "Add documentation."

    def __init__(self, exemplars_path):
        "Add documentation."
        # Define class variables:
        self.video = None
        self.frame_duration = None
        self.total_frames = None
        self.video_area = None
        self.video_secs = None
        self.window = None
        self.exemplars_set = None
        self.cmap = None
        self.exemplar_paths = None
        self.frames_handle = None

        self.exemplars_path = exemplars_path
        self.load_exemplar_paths()
        self.detector = import_module(DETECTOR).Detector()
        if USE_CLASSIFIER:
            self.classifier = import_module(
                CLASSIFIER).Classifier(self.exemplar_paths)

    def predict(self, video_filename):
        '''
        recognitions = [{
            'boxes': [] or 2D array (float32),
            'labels': [] or 1D array (int),
            'scores': [] or 1D array (float32),
            'brands': [] or 1D array (str)
            }, {}, {}, ...]
        '''
        self.set_video_source(video_filename)

        # Extract detections frame by frame:
        recognitions = []
        subclip_handle = self.video.subclip(0, self.video.end)
        for frame in tqdm(subclip_handle.iter_frames(),
                          total=self.total_frames,
                          desc='Processing video'):
            # Detect all classes:
            detections = self.detector.predict(frame)
            if USE_CLASSIFIER:
                # Select the desired classes:
                classifications = self.classifier.predict(
                    detections, frame)
                recognitions.append(classifications)
            else:
                recognitions.append(detections)
        # Draw the final detections:
        self.draw_video(recognitions)
        return self.save_video(video_filename)

    def load_exemplar_paths(self):
        "Add documentation."
        all_paths = sorted(glob.glob(os.path.join(self.exemplars_path, '*')))
        self.exemplar_paths = [path for path in all_paths
                                if get_class_name(path) in BRAND_LOGOS]
        self.exemplars_set = sorted(set(
            [get_class_name(path) for path in self.exemplar_paths]))
        self.cmap = plt.cm.get_cmap('jet', len(self.exemplars_set))
        
    def set_video_source(self, video_filename):
        "Add documentation."
        # Set source:
        self.video = VideoFileClip(video_filename)
        # Get video metadata:
        self.fps = self.video.fps
        self.frame_duration = (1 / self.fps)
        self.total_frames = int(self.video.reader.nframes)
        self.video_area = self.video.size[0] * self.video.size[1]
        self.video_secs = int(self.video.duration)

    def draw_video(self, recognitions):
        "Add documentation."
        frames_list = []
        subclip_handle = self.video.subclip(0, self.video.end)
        for idx, frame in enumerate(tqdm(subclip_handle.iter_frames(),
                                         total=self.total_frames,
                                         desc='Rendering video')):
            result = self.overlay_boxes(frame, recognitions[idx])
            frames_list.append(ImageClip(result).set_duration(1))
        # Args necessary to keep audio in all players:
        self.frames_handle = concatenate(frames_list)

    def overlay_boxes(self, image, recognitions):
        "Add documentation."
        if len(recognitions['boxes']) != 0:
            boxes = recognitions['boxes']
            scores = recognitions['scores']
            brands = recognitions['brands']
            font = cv2.FONT_HERSHEY_SIMPLEX

            for box, score, brand in zip(boxes, scores, brands):
                label = self.exemplars_set.index(brand)
                color = tuple(np.array(self.cmap(label))[:3] * 255)
                top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
                image = cv2.rectangle(image, top_left, bottom_right, color, 2)
                text = '{}: {:.2f}'.format(brand, score)
                cv2.putText(image, text, top_left, font, 0.7, color, 2)           
                
        return image

    def save_video(self, video_filename):
        # 'extension' includes the '.':
        name, extension = os.path.splitext(video_filename)
        output_filename = video_filename.replace(extension, '_output.mp4')
        self.frames_handle.write_videofile(
            output_filename,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=output_filename + '.tmp',
            remove_temp=True,
            fps=self.fps)
        return output_filename
