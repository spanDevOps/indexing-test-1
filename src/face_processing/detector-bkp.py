import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils.logging_config import get_logger

class FaceDetector:
    def __init__(self, config):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing Face Detector")
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.logger.info("Face Detector initialized successfully")
        self.config = config

    def calculate_image_properties(self, image):
        if image is None or image.size == 0:
            self.logger.warning("Empty image received in calculate_image_properties")
            return 0, 0, 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        brightness = hsv[:, :, 2].mean()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        
        brightness_percentage = (brightness / 255) * 100
        sharpness_percentage = min((sharpness / 1000) * 100, 100)
        contrast_percentage = (contrast / 255) * 100
        
        return sharpness_percentage, brightness_percentage, contrast_percentage

    def normalize_landmarks(self, landmarks, img_width, img_height):
        if landmarks is None or len(landmarks) != 5:
            self.logger.warning("Invalid landmarks detected")
            return None
        
        landmark_names = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
        return {
            landmark_names[idx]: {"X": float(value[0] / img_width), "Y": float(value[1] / img_height)}
            for idx, value in enumerate(landmarks)
        }
    
    def classify_view(self, pose):
        roll, yaw, pitch = pose
        if yaw < -60:
            return "left"
        elif yaw > 60:
            return "right"
        elif -60 <= yaw < -15:
            return "left half-profile"
        elif 15 < yaw <= 60:
            return "right half-profile"
        if -15 <= yaw <= 15:
            if pitch > 30:
                return "up"
            elif pitch < -30:
                return "down"
            else:
                return "front"
        return "uncertain"
    
    def calculate_blurriness(self, image):
        if image is None or image.size == 0:
            self.logger.warning("Empty image received in calculate_blurriness")
            return "blurry"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 100, 200)
        edge_pixel_count = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = (edge_pixel_count / total_pixels) * 100
        
        if laplacian_var < self.config["blur_threshold"] and edge_density < self.config["edge_density_threshold"]:
            return "blurry"
        return "sharp"

    def process_image(self, img_path, body):
        self.logger.info(f"Processing image: {img_path}")
        faces_data = []
        try:
            self.logger.debug(f"Body: {body}")
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image from {img_path}.")
            
            img_height, img_width, _ = image.shape
            faces = self.app.get(image)
            self.logger.info(f"Detected {len(faces)} faces in the image")
            
            for i, face in enumerate(faces):
                try:
                    if face.bbox is None or len(face.bbox) != 4:
                        self.logger.warning(f"Invalid bounding box for face {i+1}")
                        continue

                    x1, y1, x2, y2 = map(int, face.bbox)
                    face_img = image[y1:y2, x1:x2]

                    if face_img is None or face_img.size == 0:
                        self.logger.warning(f"Empty face image for face {i+1}")
                        continue

                    sharpness, brightness, contrast = self.calculate_image_properties(face_img)
                    quality_score = float(face.det_score)

                    if sharpness < self.config["sharpness_threshold"]:
                        self.logger.info(f"Face skipped due to low sharpness in {img_path}")
                        continue

                    if brightness < self.config["brightness_threshold"]:
                        self.logger.info(f"Face skipped due to low brightness in {img_path}")
                        continue

                    if quality_score < self.config["face_confidence_threshold"]:
                        self.logger.info(f"Face skipped due to low confidence in {img_path}")
                        continue

                    if self.calculate_blurriness(face_img) == "blurry":
                        self.logger.info(f"Face skipped due to blur in {img_path}")
                        continue

                    landmarks = self.normalize_landmarks(face.kps, img_width, img_height)
                    if landmarks is None:
                        self.logger.warning(f"Could not normalize landmarks for face {i+1}")
                        continue

                    view = self.classify_view(face.pose)

                    face_data = {
                        "boundingBox": {
                            "Width": float((x2 - x1) / img_width),
                            "Height": float((y2 - y1) / img_height),
                            "Left": float(x1 / img_width),
                            "Top": float(y1 / img_height),
                        },
                        "landmarks": landmarks,
                        "embedding": face.embedding.tolist() if face.embedding is not None else None,
                        "info": {
                            "age": int(face.age) if face.age is not None else None,
                            "gender": 'male' if face.gender == 1 else 'female' if face.gender == 0 else None,
                        },
                        "pose": {
                            **{k: float(v) for k, v in zip(["Roll", "Yaw", "Pitch"], face.pose) if v is not None},
                            "view": view
                        },
                        "quality": {
                            "Brightness": float(brightness),
                            "Sharpness": float(sharpness),
                            "Contrast": float(contrast),
                            "quality_score": float(quality_score)
                        },
                    }
                    
                    faces_data.append(face_data)

                except Exception as e:
                    self.logger.error(f"Error processing face {i+1}: {str(e)}", exc_info=True)

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}", exc_info=True)

        self.logger.info(f"Processed {len(faces_data)} faces.")
        return faces_data