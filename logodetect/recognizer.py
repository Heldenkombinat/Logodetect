import os
import glob
from importlib import import_module

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import (
    VideoFileClip,
    VideoClip,
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)

from logodetect.utils import clean_name
from logodetect.constants import (
    DETECTOR,
    CLASSIFIER,
    USE_CLASSIFIER,
    BRAND_LOGOS,
    EXEMPLARS_FORMAT,
)


class Recognizer(object):
    """Recognizer

    The Recognizer is the main class for detecting and recognizing logos.
    """

    def __init__(self, exemplars_path: str):
        self.video = None
        self.audio = None
        self.frame_duration = None
        self.total_frames = None
        self.exemplars_path = exemplars_path
        all_paths = glob.glob(
            os.path.join(self.exemplars_path, "*.{}".format(EXEMPLARS_FORMAT))
        )
        self.exemplar_paths = [
            path for path in all_paths if clean_name(path) in BRAND_LOGOS
        ]
        self.exemplars_set = sorted(
            set([clean_name(path) for path in self.exemplar_paths])
        )
        self.cmap = plt.cm.get_cmap("jet", len(self.exemplars_set))
        self.detector = import_module(f"logodetect.{DETECTOR}").Detector()
        if USE_CLASSIFIER:
            self.classifier = import_module(
                f"logodetect.classifiers.{CLASSIFIER}"
            ).Classifier(self.exemplar_paths)

    def predict(self, video_filename: str, output_appendix: str = "_output") -> None:
        """ Predict recognitions, which will come in the following form:

        recognitions = [{
            'boxes': [] or 2D array (float32),
            'labels': [] or 1D array (int),
            'scores': [] or 1D array (float32),
            'brands': [] or 1D array (str)
            }, {}, {}, ...]

        :param video_filename: file name of the video to process
        :param output_appendix: this string will be appended to the name of the processed video
        :return: None, processed video will be saved
        """
        self.set_video_source(video_filename)

        recognitions = []
        sub_clip_handle = self.video.subclip(0, self.video.end)
        for frame in tqdm(
            sub_clip_handle.iter_frames(),
            total=self.total_frames,
            desc="Processing video",
        ):
            recognitions = self.compute_recognitions(frame, recognitions)
        predicted_video = self.draw_video(recognitions)
        self.save_video(predicted_video, video_filename, output_appendix)

    def predict_image(
        self, image_filename: str, output_appendix: str = "_output"
    ) -> None:
        """ Predict recognitions for a single input image.

        :param image_filename: file name of the image to process
        :param output_appendix: this string will be appended to the name of the processed image
        :return: None, processed image will be saved
        """
        image = cv2.imread(image_filename)
        recognitions = self.compute_recognitions(image)
        predicted_image = self.draw_overlay_boxes(image, recognitions[0])
        self.save_image(predicted_image, image_filename, output_appendix)

    def compute_recognitions(self, frame: np.ndarray, recognitions: list = None):
        """Compute new recognitions for this frame and returns the augmented list.

        :param frame: current frame
        :param recognitions: previous list of recognitions
        :return: updated list of recognitions
        """
        if not recognitions:
            recognitions = []
        detections = self.detector.predict(frame)
        if USE_CLASSIFIER:
            classifications = self.classifier.predict(detections, frame)
            recognitions.append(classifications)
        else:
            recognitions.append(detections)
        return recognitions

    def set_video_source(self, video_filename) -> None:
        """Set all video related properties for the specified video

        :param video_filename: file name of the video to process
        :return: None
        """
        self.video = VideoFileClip(video_filename)
        self.audio = AudioFileClip(video_filename)
        self.total_frames = int(self.video.reader.nframes)
        self.frame_duration = 1 / self.video.fps

    def draw_video(self, recognitions: list) -> VideoClip:
        """Draw the recognitions found for the video into the video frames
        and store them in "all_frames".

        :param recognitions: list of recognitions.
        :return:
        """
        all_frames = []
        sub_clip_handle = self.video.subclip(0, self.video.end)
        for idx, frame in enumerate(
            tqdm(
                sub_clip_handle.iter_frames(),
                total=self.total_frames,
                desc="Rendering video",
            )
        ):
            result = self.draw_overlay_boxes(frame, recognitions[idx])
            new_frame = ImageClip(result).set_duration(self.frame_duration)
            all_frames.append(new_frame)
        return concatenate_videoclips(all_frames)

    def draw_overlay_boxes(self, image, recognition: dict):
        """If there a recognitions for this image, draw overlay
        boxes and text on the image.

        :param image: input image
        :param recognition: recognition object
        :return: processed image with overlay boxes
        """
        if len(recognition["boxes"]) != 0:
            boxes = recognition["boxes"]
            scores = recognition["scores"]
            brands = recognition["brands"]
            font = cv2.FONT_HERSHEY_SIMPLEX

            for box, score, brand in zip(boxes, scores, brands):
                label = self.exemplars_set.index(brand)
                color = tuple(np.array(self.cmap(label))[:3] * 255)
                top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
                image = cv2.rectangle(image, top_left, bottom_right, color, 2)
                text = "{}: {:.2f}".format(brand, score)
                cv2.putText(image, text, top_left, font, 0.7, color, 2)

        return image

    @staticmethod
    def save_image(
        image: np.ndarray, image_filename: str, output_appendix: str = "_output"
    ) -> None:
        """Save the resulting image.

        :param image: processed image
        :param image_filename: original file name
        :param output_appendix: appendix to add to file
        :return: None
        """
        name, extension = os.path.splitext(image_filename)
        output_filename = image_filename.replace(
            extension, f"{output_appendix}{extension}"
        )
        print(f"Saved resulting image as {output_filename}.")
        cv2.imwrite(output_filename, image)

    def save_video(
        self, video: VideoClip, video_filename: str, output_appendix: str = "_output"
    ):
        """Save the resulting video.

        :param video: the processed VideoClip
        :param video_filename: original file name
        :param output_appendix: appendix to add to file
        :return: None
        """
        name, extension = os.path.splitext(video_filename)
        output_filename = video_filename.replace(
            extension, f"{output_appendix}{extension}"
        )
        video.set_audio(self.audio)
        video.write_videofile(
            output_filename,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=output_filename + ".tmp",
            remove_temp=True,
            fps=self.video.fps,
        )
