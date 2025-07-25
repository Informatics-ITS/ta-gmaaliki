from tracking.abstract_tracker import AbstractTracker

import cv2
import supervision as sv
from typing import List
from ultralytics.engine.results import Results
import numpy as np

class KeypointsTracker(AbstractTracker):
    """Detection and Tracking of football field keypoints"""

    def __init__(self, model_path: str, conf: float = 0.1, kp_conf: float = 0.7, det_size: int = 1280) -> None:
        """
        Initialize KeypointsTracker for tracking keypoints.
        
        Args:
            model_path (str): Model path.
            conf (float): Confidence threshold for field detection.
            kp_conf (float): Confidence threshold for keypoints.
        """
        super().__init__(model_path, conf)  # Call the Tracker base class constructor
        self.kp_conf = kp_conf  # Keypoint Confidence Threshold
        self.tracks = []  # Initialize tracks list
        self.cur_frame = 0  # Frame counter initialization
        self.original_size = (1920, 1080)  # Original resolution (1920x1080)
        self.det_size = det_size
        self.scale_x = self.original_size[0] / self.det_size
        self.scale_y = self.original_size[1] / self.det_size

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Perform keypoint detection on multiple frames.

        Args:
            frames (List[np.ndarray]): List of frames for detection.
        
        Returns:
            List[Results]: Detected keypoints for each frame
        """
        # Adjust contrast before detection for each frame
        contrast_adjusted_frames = [self._preprocess_frame(frame) for frame in frames]

        # Use YOLOv8's batch predict method
        detections = self.model.predict(contrast_adjusted_frames, conf=self.conf, verbose=False)

        top_detections = []

        for result in detections:
            # Default values
            best_conf = -1
            best_idx = None

            if result.boxes is not None and len(result.boxes.conf) > 0:
                for i, conf in enumerate(result.boxes.conf):
                    if conf > best_conf:
                        best_conf = conf
                        best_idx = i

            if best_idx is not None:
                # Create single-detection result
                top_result = Results(
                    orig_img=result.orig_img,
                    path=result.path,
                    names=result.names
                )
                top_result.boxes = result.boxes[best_idx:best_idx + 1]
                top_result.keypoints = result.keypoints[best_idx:best_idx + 1] if result.keypoints is not None else None
                top_result.probs = result.probs
                top_result.speed = result.speed

                top_detections.append(top_result)
            else:
                # Optionally append empty result if no detections
                top_detections.append(Results(
                    orig_img=result.orig_img,
                    path=result.path,
                    names=result.names
                    ))

        return top_detections

    def track(self, detection: Results) -> dict:
        """
        Perform keypoint tracking based on detections.
        
        Args:
            detection (Results): Detected keypoints for a single frame.
        
        Returns:
            dict: Dictionary containing tracks of the frame.
        """
        # Check 
        if detection.boxes is None:
            return {}

        detection = sv.KeyPoints.from_ultralytics(detection)

        # Extract xy coordinates, confidence, and the number of keypoints
        xy = detection.xy[0]  # Shape: (32, 2), assuming there are 32 keypoints
        confidence = detection.confidence[0]  # Shape: (32,), confidence values

        # Create the map of keypoints with confidence greater than the threshold
        filtered_keypoints = {
            i: (coords[0] * self.scale_x, coords[1] * self.scale_y)  # i is the key (index), (x, y) are the values
            for i, (coords, conf) in enumerate(zip(xy, confidence))
            if conf > self.kp_conf
            and 0 <= coords[0] <= self.det_size  # Check if x is within bounds
            and 0 <= coords[1] <= self.det_size  # Check if y is within bounds
        }

        self.tracks.append(detection)
        self.cur_frame += 1

        return filtered_keypoints

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame by adjusting contrast and resizing to original detection size.
        
        Args:
            frame (np.ndarray): The input image frame.
        
        Returns:
            np.ndarray: The resized frame with adjusted contrast.
        """
        # Adjust contrast
        # NOTE: contrast will NOT work on the current model
        # frame = self._adjust_contrast(frame)
        
        # Resize frame to original detection size
        resized_frame = cv2.resize(frame, (self.det_size, self.det_size))

        return resized_frame
    
    def _adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Adjust the contrast of the frame using Histogram Equalization.
        
        Args:
            frame (np.ndarray): The input image frame.
        
        Returns:
            np.ndarray: The frame with adjusted contrast.
        """
        # Check if the frame is colored (3 channels). If so, convert to grayscale for histogram equalization.
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to the Y channel (luminance)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR format
            frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # If the frame is already grayscale, apply histogram equalization directly
            frame_equalized = cv2.equalizeHist(frame)

        return frame_equalized
