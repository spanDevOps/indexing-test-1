import os
import cv2
import numpy as np
import torch
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from scipy import stats
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, List, Union
from pathlib import Path
import warnings
import traceback

class FaceDetector:
    def __init__(self, config: Dict[str, Any]):
        try:
            print("Starting FaceDetector initialization...")
            
            print("Setting up CUDA providers...")
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            print(f"Using providers: {self.providers}")
            
            print("Initializing face_app...")
            self.face_app = FaceAnalysis(providers=self.providers)
            print("Preparing face_app...")
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("Face_app prepared")
            
            # Create models directory in user's home
            home_dir = str(Path.home())
            insightface_dir = os.path.join(home_dir, '.insightface', 'models')
            os.makedirs(insightface_dir, exist_ok=True)
            
            print(f"Using InsightFace models directory: {insightface_dir}")
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
            
            if self.device == 'cuda':
                self.providers = ['CUDAExecutionProvider']
                torch.cuda.init()
                print(f"CUDA initialized:")
                print(f"- Device count: {torch.cuda.device_count()}")
                print(f"- Device name: {torch.cuda.get_device_name(0)}")
                print(f"- CUDA version: {torch.version.cuda}")
            else:
                self.providers = ['CPUExecutionProvider']

            # Get the absolute path to the models directory
            current_dir = Path(__file__).parent.parent
            models_dir = current_dir / "models"
            
            # Check and load models
            face_occlusion_path = models_dir / "face_occlusion.onnx"
            webface_path = models_dir / "webface_r50_pfc.onnx"
            
            if not face_occlusion_path.exists():
                raise FileNotFoundError(f"Model not found at {face_occlusion_path}")
            if not webface_path.exists():
                raise FileNotFoundError(f"Model not found at {webface_path}")
            
            self.occlu_model = onnxruntime.InferenceSession(
                str(face_occlusion_path), 
                providers=self.providers
            )
            self.embedding_model = onnxruntime.InferenceSession(
                str(webface_path), 
                providers=self.providers
            )

            self.preprocess_embedding = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            self.preprocess_occlusion = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            print("FaceDetector initialization completed")
            
        except Exception as e:
            print("Initialization error")
            print(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize FaceDetector: {str(e)}")

    def process_image(self, image_path: Union[str, Path], body: Dict[str, Any]) -> List[Dict]:
        print(f"image_path: {image_path}")
        faces_data = []
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Failed to load image from path: {image_path}")
                return []  # Return empty list instead of error string
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            
            if not faces:
                print("No faces detected in image")
                return []  # Return empty list instead of error string
            
            for face in faces:
                aligned_face = face_align.norm_crop(img_rgb, face.kps.astype(int), image_size=112)
                
                blur_result = self._analyze_blur(aligned_face)
                if blur_result["quality"] == "blurry":
                    continue
                
                face_result = self._process_face(aligned_face)
                x1, y1, x2, y2 = map(int, face.bbox)

                # Handle potentially missing or None attributes
                age = face.age if hasattr(face, 'age') and face.age is not None else 0
                gender = face.gender if hasattr(face, 'gender') and face.gender is not None else 0
                det_score = face.det_score if hasattr(face, 'det_score') and face.det_score is not None else 0.0

                face_data = {
                    "boundingBox": {
                        "Width": float((x2 - x1) / img.shape[1]),
                        "Height": float((y2 - y1) / img.shape[0]),
                        "Left": float(x1 / img.shape[1]),
                        "Top": float(y1 / img.shape[0]),
                    },
                    "landmarks": {
                        "points": face.kps.astype(float).tolist(),
                        "validity": self._check_landmarks_validity(face.kps, img_rgb.shape)
                    },
                    "orientation": self._analyze_orientation(face),
                    "info": {
                        "age": int(age),
                        "gender": "male" if gender == 1 else "female",
                        "gender_confidence": float(det_score)
                    },
                    "confidence": float(det_score),
                    "embedding": face_result["embedding"].tolist()[0],
                    "quality": face_result["quality"],
                }
                faces_data.append(face_data)
            
            return faces_data
            
        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return []  # Return empty list instead of error string

    def _process_face(self, face_img: np.ndarray) -> Dict[str, Any]:
        # Your existing _process_face implementation
        aligned_face_pil = Image.fromarray(face_img)
        face_tensor_embedding = self.preprocess_embedding(aligned_face_pil).unsqueeze(0)
        face_tensor_occlusion = self.preprocess_occlusion(aligned_face_pil).unsqueeze(0)
        
        embedding = self._get_embedding(face_tensor_embedding.numpy())
        blur_result = self._analyze_blur(face_img)
        illumination_result = self._analyze_illumination(face_img)
        occlusion_result = self._detect_occlusion(face_tensor_occlusion.numpy())
        
        return {
            "embedding": embedding,
            "quality": {
                "blur": blur_result,
                "illumination": illumination_result,
                "occlusion": occlusion_result
            }
        }

    def _get_embedding(self, face_tensor: np.ndarray) -> np.ndarray:
        # Your existing implementation
        input_name = self.embedding_model.get_inputs()[0].name
        embedding = self.embedding_model.run([self.embedding_model.get_outputs()[0].name], {input_name: face_tensor})[0]
        return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

    def _analyze_blur(self, face_img: np.ndarray) -> Dict[str, Any]:
        # Your existing implementation
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        h, w = gray.shape
        high_freq_mask = np.zeros_like(gray, dtype=bool)
        high_freq_mask[h//4:3*h//4, w//4:3*w//4] = True
        high_freq_energy = np.sum(np.abs(fshift[high_freq_mask]))
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        metrics = {
            'laplacian_var': float(laplacian_var),
            'high_freq_energy': float(high_freq_energy),
            'edge_density': float(edge_density)
        }
        
        quality = 'clear' if laplacian_var > 50 and high_freq_energy > 1000 and edge_density > 0.05 else 'blurry'
        return {"quality": quality, "metrics": metrics}

    def _analyze_illumination(self, face_img: np.ndarray) -> Dict[str, Any]:
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        l_channel = cv2.split(lab)[0]
        
        dark_ratio = np.sum(l_channel < 64) / l_channel.size * 100
        bright_ratio = np.sum(l_channel > 192) / l_channel.size * 100
        dynamic_range = float(np.max(l_channel) - np.min(l_channel))
        
        metrics = {
            'dark_ratio': float(dark_ratio),
            'bright_ratio': float(bright_ratio),
            'dynamic_range': dynamic_range
        }
        
        if dark_ratio > 30:
            quality = 'underexposed'
        elif bright_ratio > 30:
            quality = 'overexposed'
        elif dynamic_range < 40:
            quality = 'low_contrast'
        else:
            quality = 'good'
            
        return {"quality": quality, "metrics": metrics}

    def _detect_occlusion(self, face_tensor: np.ndarray) -> Dict[str, Any]:
        input_name = self.occlu_model.get_inputs()[0].name
        output = self.occlu_model.run(None, {input_name: face_tensor})[0]
        prob = torch.softmax(torch.from_numpy(output), 1)
        confidence, prediction = torch.max(prob, 1)
        
        return {
            "occluded": bool(prediction.item()),
            "confidence": float(confidence.item() * 100)
        }

    def _analyze_orientation(self, face) -> Dict[str, Any]:
        if not hasattr(face, 'pose') or face.pose is None:
            return {
                "orientation": "Front",
                "percentage": 100.0
            }
            
        pitch, yaw, roll = face.pose
        yaw_percentage = float(min(100, abs(yaw / 70 * 100)))
        pitch_percentage = float(min(100, abs(pitch / 60 * 100)))
        front_yaw = float(100 * (1 - (abs(yaw) / 70)**2))
        front_pitch = float(100 * (1 - (abs(pitch) / 60)**2))
        
        if abs(yaw) > abs(pitch):
            if yaw > 30:
                orientation = "Side Left"
                percentage = yaw_percentage
            elif yaw < -30:
                orientation = "Side Right"
                percentage = yaw_percentage
            else:
                orientation = "Front"
                percentage = front_yaw
        else:
            if pitch > 20:
                orientation = "Up"
                percentage = pitch_percentage
            elif pitch < -20:
                orientation = "Down"
                percentage = pitch_percentage
            else:
                orientation = "Front"
                percentage = front_pitch
        
        return {
            "orientation": orientation,
            "percentage": float(round(percentage, 2))
        }

    def _check_landmarks_validity(self, kps: np.ndarray, image_shape: tuple) -> Dict[str, Any]:
        height, width = image_shape[:2]
        total_points = len(kps)
        valid_points = 0
        
        for x, y in kps:
            if 0 <= x <= width and 0 <= y <= height:
                valid_points += 1
        
        validity_percentage = (valid_points / total_points) * 100
        return {
            "valid_percentage": float(round(validity_percentage, 2)),
            "valid_points": valid_points,
            "total_points": total_points,
            "all_valid": valid_points == total_points
        }