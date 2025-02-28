import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils.logging_config import get_logger
from skimage import img_as_float

class FaceDetector:
    def __init__(self, config):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing Face Detector")
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)
        self.logger.info("Face Detector initialized successfully")
        self.config = config



    def determine_pose_view(self, pose):
        roll, yaw, pitch = pose

        if abs(pitch) < 10 and abs(yaw) < 10 and abs(roll) < 10:
            return "front"

        if yaw >= 30:
            return "right profile"
        elif yaw <= -30:
            return "left profile"

        if 10 <= yaw < 30:
            return "right half-profile"
        elif -30 < yaw <= -10:
            return "left half-profile"

        if pitch > 20:
            return "looking up"
        elif pitch < -20:
            return "looking down"

        if roll > 20:
            return "tilted right"
        elif roll < -20:
            return "tilted left"
        return "unknown"


    def calculate_image_attributes(self, image):
        if image.size == 0:
            raise ValueError("Image is empty.")
        image_float = img_as_float(image)
        sharpness = cv2.Laplacian(image_float, cv2.CV_64F).var()
        contrast = image_float.std() * 100
        brightness = np.mean(image_float) * 100
        return sharpness, contrast, brightness



    def process_image(self, img_path, body):

        self.logger.info(f"Processing image: {img_path}")

        image = cv2.imread(img_path)
    
        if image is None:
            print(f"Error: Image at {img_path} cannot be read.")
            return [], []

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_image)
        
        faces_data = []
        try:
            self.logger.debug(f"Body: {body}")
            for idx, face in enumerate(faces):
                det_score = face.det_score
                if det_score < 0.65: 
                    continue
                
                bbox = face.bbox.astype(int)
                if (bbox[0] < 0 or bbox[1] < 0 or 
                    bbox[2] > image.shape[1] or bbox[3] > image.shape[0] or 
                    bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                    print(f"Warning: Invalid bounding box for face in image {img_path}. Skipping.")
                    continue
                
                face_image = rgb_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if face_image.size == 0:
                    print(f"Warning: Extracted face image is empty for {img_path}. Skipping.")
                    continue

                sharpness, contrast, brightness = self.calculate_image_attributes(face_image)

                embedding = face.embedding
                normalized_embedding = embedding / np.linalg.norm(embedding)
                key_landmarks_indices = [36, 39, 42, 45, 30, 48, 54, 8]
                selected_landmarks = face.landmark_2d_106[key_landmarks_indices]
                normalized_landmarks = (selected_landmarks - np.mean(selected_landmarks, axis=0)) / np.std(selected_landmarks, axis=0)
                landmark_features = normalized_landmarks.flatten()
                pose_info = face.pose / np.linalg.norm(face.pose)

                faceprint = np.concatenate((normalized_embedding * 0.74,
                                            landmark_features * 0.21,
                                            pose_info * 0.05))

                
                
                
                face_data = {
                    "embedding": face.embedding.tolist() if face.embedding is not None else None,
                    "boundingBox": {
                        "Width": float((bbox[2] - bbox[0]) / image.shape[1]),
                        "Height": float((bbox[3] - bbox[1]) / image.shape[0]),
                        "Left": float(bbox[0] / image.shape[1]),
                        "Top": float(bbox[1] / image.shape[0]),
                    },
                    "info": {
                        "age": int(face.age),
                        "gender": 'male' if face.gender == 1 else 'female' if face.gender == 0 else None,
                    },
                    "landmarks": [
                        {
                            "Type": f"landmark_{i}",
                            "X": float(coords[0]),
                            "Y": float(coords[1])
                        }
                        for i, (landmark_type, coords) in enumerate(zip(["left_eye_left_corner", "left_eye_right_corner", "right_eye_left_corner", "right_eye_right_corner", "nose_tip", "mouth_left_corner", "mouth_right_corner", "chin_bottom"], selected_landmarks))
                    ],
                    "pose": {
                        "Roll": float(face.pose[0]),
                        "Yaw": float(face.pose[1]),
                        "Pitch": float(face.pose[2]),
                        "View": self.determine_pose_view(face.pose),
                    },
                    "quality": {
                        "Brightness": float(brightness),
                        "Sharpness": float(sharpness),
                        "Contrast": float(contrast),
                    },
                    "confidence": float(det_score),
                    "faceprint": faceprint.tolist(),
                }
                faces_data.append(face_data)

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}", exc_info=True)

        self.logger.info(f"Processed {len(faces_data)} faces.")
        return faces_data