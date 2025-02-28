from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
import time
from typing import Dict
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass, asdict
from collections.abc import Mapping



print("Starting mongodb_handler.py")

@dataclass
class FaceData(Mapping):
    orientation: str
    orientation_percentage: float
    face_confidence: float
    embedding: np.ndarray
    _id: str = None
    image_id: str = None
    optimizedImageS3Key: str = None
    boundingBox: dict = None
    face_id: str = None
    album_id: str = None
    imageVersionId: str = None
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        """Mimics dictionary get() method"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default
    
    def __iter__(self):
        """Required for Mapping interface"""
        return iter(self.__dict__)
    
    def __len__(self):
        """Required for Mapping interface"""
        return len(self.__dict__)
    
    def keys(self):
        """Returns all keys"""
        return self.__dict__.keys()
    
    def items(self):
        """Returns all items"""
        return self.__dict__.items()
    
    def values(self):
        """Returns all values"""
        return self.__dict__.values()
    
    def to_dict(self):
        """Convert to dictionary with safe string conversion"""
        def safe_str(val):
            if isinstance(val, ObjectId):
                return str(val)
            if isinstance(val, str):
                return val
            if val is None:
                return None
            return str(val)
            
        return {
            "_id": safe_str(self._id),
            "image_id": safe_str(self.image_id),
            "optimizedImageS3Key": self.optimizedImageS3Key,
            "boundingBox": self.boundingBox,
            "face_id": safe_str(self.face_id),
            "orientation": self.orientation,
            "orientation_percentage": float(self.orientation_percentage),
            "face_confidence": float(self.face_confidence),
            "album_id": safe_str(self.album_id),
            "imageVersionId": self.imageVersionId
        }

class MongoDBHandler:
    def __init__(self, gallery_images_uri, huemn_uri):
        print("Initializing MongoDBHandler")
        self.gallery_images_client = MongoClient(gallery_images_uri)
        self.huemn_client = MongoClient(huemn_uri)
        self.gallery_faces_collection = self.gallery_images_client['production']['galleryFaces']
        self.gallery_images_collection = self.huemn_client['production']['galleryimages']
        self.gallery_upload_batch_collection = self.huemn_client['production']['galleryuploadbatch']
        self.failed_messages_collection = self.gallery_images_client['production']['failedmessages']
        self.gallery_image_faces_collection = self.huemn_client['production']['galleryimagefaces']
        print("MongoDB Handler initialized")



    def create_face_document(self, face_data: Dict, body: Dict) -> ObjectId:
        """Create a face document in MongoDB"""
        try:
            # Create a new dictionary with all the required fields
            face_doc = {
                'tenant_id': ObjectId(body.get('tenantId')),
                'uploadBatchId': body.get('uploadBatchId'),
                'gallery_id': ObjectId(body.get('gallery_id')),
                'album_id': ObjectId(body.get('album_id')),
                'image_id': ObjectId(body.get('image_id')),
                'imageVersionId': body.get('imageVersionId'),
                'optimizedImageS3Key': body.get('optimizedImageS3Key'),
                'createdAt': int(time.time()),
                'updatedAt': int(time.time()),
                # Add all the face detection data
                'boundingBox': face_data.get('boundingBox', {}),
                'landmarks': face_data.get('landmarks', {}),
                'orientation': face_data.get('orientation', {}),
                'info': face_data.get('info', {}),
                'confidence': face_data.get('confidence', 0.0),
                'embedding': face_data.get('embedding', []),
                'quality': face_data.get('quality', {})
            }

            result = self.gallery_faces_collection.insert_one(face_doc)
            return result.inserted_id
            
        except Exception as e:
            print(f"Error creating face document: {str(e)}")
            print(f"Face data: {face_data}")
            print(f"Body: {body}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
        
    def filter_face_data(self, face):
        return {
            "boundingBox": face.get("boundingBox", {}),
            "landmarks": face.get("landmarks", {}),
            "info": face.get("info", {}),
            "pose": face.get("pose", {}),
            "quality": face.get("quality", {}),
            "ec2FaceId": ObjectId(face.get("ec2FaceId", {})),
            "face_id": None,

        }

    def add_faces_to_image(self, image_id, version_id, faces):
        try:
            current_unix_timestamp = int(time.time())

            if 'galleryimages' not in self.huemn_client['production'].list_collection_names():
                print("Collection 'galleryimages' does not exist in the database.")
                return False

            existing_doc = self.gallery_images_collection.find_one({"_id": ObjectId(image_id)})
            if not existing_doc:
                print(f"Document with id {image_id} not found.")
                return False
            
            version_exists = any(v.get('versionId') == version_id for v in existing_doc.get('versions', []))
            if not version_exists:
                print(f"Version {version_id} not found in document {image_id}")
                return False
            
            # Filter the faces data to include only the specified fields
            filtered_faces = [self.filter_face_data(face) for face in faces]

            update_query = {
                "filter": {
                    "_id": ObjectId(image_id)
                },
                "update": {
                    "$set": {
                        "activeVersion.faces": filtered_faces,
                        "versions.$[version].faces": filtered_faces,
                        "face_ids": [],
                        "updatedAt": current_unix_timestamp
                    }
                },
                "array_filters": [
                    {
                        "version.versionId": version_id
                    }
                ]
            }

            result = self.gallery_images_collection.update_one(
                filter=update_query["filter"],
                update=update_query["update"],
                array_filters=update_query["array_filters"]
            )

            if result.modified_count > 0:
                print(f"Successfully updated faces for image {image_id}, version {version_id}")
                return True
            else:
                print(f"No documents were modified for image {image_id}, version {version_id}. The image might not exist or the version might not match.")
                return False

        except PyMongoError as e:
            print(f"MongoDB error occurred while updating faces for image {image_id}, version {version_id}: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error occurred while updating faces for image {image_id}, version {version_id}: {str(e)}")
            return False

    def update_gallery_upload_batch_counter(self, uploadBatchId):
        print(f"Updating gallery upload batch counter for {uploadBatchId}")
        try:
            self.gallery_upload_batch_collection.update_one(
                {"uploadBatchId": uploadBatchId},
                {
                    "$inc": {"indexedCount": 1},
                    "$set": {
                        "updatedAt": int(time.time()),
                        "status": "IN_PROGRESS"
                    }
                }
            )
        except PyMongoError as e:
            print(f"MongoDB error occurred while updating gallery upload batch counter: {str(e)}")
            return False

    def record_failed_message(self, failure_record):
        try:
            failure_record['type'] = 'face_index'
            self.failed_messages_collection.insert_one(failure_record)
        except PyMongoError as e:
            print(f"MongoDB error occurred while recording failed message: {str(e)}")
            return False


    def get_faces_by_gallery_id(self, gallery_id):
        print(f"Fetching faces for gallery_id: {gallery_id}")
        try:

            # First, get the count and set up pagination
            total_count = self.gallery_faces_collection.count_documents({"gallery_id": ObjectId(gallery_id)})
            if total_count == 0:
                print("No faces found for this gallery")
                return {'all_faces': [], 'specific_face': None}

            PAGE_SIZE = 10000
            total_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE

            # Base pipeline for optimization
            base_pipeline = [
                {
                    "$match": {
                        "gallery_id": ObjectId(gallery_id)
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "image_id": 1,
                        "face_id": 1,
                        "optimizedImageS3Key": 1,
                        "boundingBox": 1,
                        "embedding": 1,
                        "orientation": 1,
                        "confidence": 1,
                        "album_id": 1,
                        "imageVersionId": 1,
                    }
                }
            ]

            all_faces = []

            def process_page(page_num):
                try:
                    pipeline = base_pipeline + [
                        {"$skip": page_num * PAGE_SIZE},
                        {"$limit": PAGE_SIZE}
                    ]
                    
                    page_faces = []
                    
                    cursor = self.gallery_faces_collection.aggregate(
                        pipeline,
                        allowDiskUse=True
                    )

                    for doc in cursor:
                        try:
                            # Create a FaceData object with the required fields
                            face_data = FaceData(
                                orientation=doc['orientation']['orientation'],
                                orientation_percentage=doc['orientation']['percentage'],
                                face_confidence=doc['confidence'],
                                embedding=np.array(doc['embedding'], dtype=np.float32)
                            )
                            
                            # Add additional metadata
                            face_data._id = doc['_id']
                            face_data.image_id = doc['image_id']
                            face_data.optimizedImageS3Key = doc['optimizedImageS3Key']
                            face_data.boundingBox = doc['boundingBox']
                            face_data.face_id = doc['face_id'] if 'face_id' in doc else None
                            face_data.album_id = doc['album_id']
                            face_data.imageVersionId = doc['imageVersionId']
                            
                            page_faces.append(face_data)

                        except Exception as e:
                            print(f"Error processing document: {e}")
                            continue

                    return page_faces
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    return [], None

            # Process pages in parallel with error handling
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_page, page) for page in range(total_pages)]
                
                for i, future in enumerate(futures):
                    try:
                        page_faces = future.result()
                        all_faces.extend(page_faces)
                        print(f"Processed page {i + 1}/{total_pages}")
                    except Exception as e:
                        print(f"Error processing future {i}: {e}")
                        continue

            return {
                'all_faces': all_faces,
            }

        except Exception as e:
            print(f"An error occurred while fetching faces: {str(e)}")
            return {'all_faces': [], 'specific_face': None}

    def get_face_id(self, image_id, ec2_face_id):
        print(f"Getting face_id for image_id: {image_id}, ec2_face_id: {ec2_face_id}")
        gallery_image_doc = self.gallery_images_collection.find_one(
            {
                '_id': ObjectId(image_id)
            },
            {
                'versions.faces.ec2FaceId': 1,
                'versions.faces.face_id': 1
            }
        )

        if not gallery_image_doc:
            print(f"Image not found for image_id: {image_id}")
            return {
                'imageExists': False,
                'face_id': None
            }

        faces = []
        for version in gallery_image_doc.get('versions', []):
            if version.get('faces'):
                faces.extend(version['faces'])

        matching_face = next((face for face in faces if face['ec2FaceId'] == ec2_face_id), None)

        if matching_face:
            print(f"Found matching face with face_id: {matching_face.get('face_id')}")
            return {
                'imageExists': True,
                'face_id': matching_face.get('face_id')
            }
        else:
            print(f"No matching face found for ec2_face_id: {ec2_face_id}")
            return {
                'imageExists': True,
                'face_id': None
            }

    def update_face_id_for_image_face(self, image_id, image_version_id, ec2_face_id, face_id):
        print(f"Updating face_id for image: {image_id}, ec2_face_id: {ec2_face_id}")
        try:
            # Convert string IDs to ObjectId if necessary
            image_id_obj = ObjectId(image_id) if isinstance(image_id, str) else image_id
            face_id_obj = ObjectId(face_id) if isinstance(face_id, str) else face_id

            print(f"face_id_obj: {face_id_obj}")

            current_unix_timestamp = int(time.time())

            # Fetch the document first to check its structure
            document = self.gallery_images_collection.find_one({'_id': image_id_obj})
            if not document:
                print(f"No document found with _id: {image_id}")
                return

            print(f"Document structure: {document}")

            # Construct the update query
            update_query = {
                '$set': {
                    'activeVersion.faces.$[face].face_id': face_id_obj,
                    'versions.$[version].faces.$[face].face_id': face_id_obj,
                    'updatedAt': current_unix_timestamp
                },
                '$addToSet': {
                    'face_ids': face_id_obj
                }
            }

            # Construct the array filters
            array_filters = [
                {"version.versionId": image_version_id},
                {"face.ec2FaceId": ec2_face_id}  # Changed from awsFaceId to ec2FaceId
            ]

            # Perform the update
            result = self.gallery_images_collection.update_one(
                {'_id': image_id_obj},
                update_query,
                array_filters=array_filters
            )

            print(f"Update result: {result.matched_count} document(s) matched, {result.modified_count} document(s) modified")

            if result.matched_count == 0:
                print("No matching document found. Check your query conditions.")
            elif result.modified_count == 0:
                print("Document matched but not modified. Check if the update is necessary.")

        except Exception as e:
            print(f"An error occurred while updating face_id: {str(e)}")
            raise


    def create_gallery_image_face(self, tenant_id, gallery_id, display_image, id_maps):
        print(f"Creating gallery image face for gallery_id: {gallery_id}")
        current_unix_timestamp = int(time.time())

        # Check for duplicate ec2FaceId in id_maps
        ec2_face_ids = set()
        unique_id_maps = []
        for id_map in id_maps:
            ec2_face_id = id_map.get('ec2FaceId')
            if ec2_face_id not in ec2_face_ids:
                ec2_face_ids.add(ec2_face_id)
                unique_id_maps.append(id_map)
            else:
                print(f"Duplicate ec2FaceId found: {ec2_face_id}. Skipping this entry.")
        
        # Replace the original id_maps with the deduplicated list
        id_maps = unique_id_maps

        document = {
            "tenant_id": ObjectId(tenant_id),
            "gallery_id": ObjectId(gallery_id),
            "name": None, 
            "displayImage": display_image,
            "idMaps": id_maps,
            "createdAt": current_unix_timestamp,
            "updatedAt": current_unix_timestamp 

        }
        result = self.gallery_image_faces_collection.insert_one(document)
        print(f"Created gallery image face with _id: {result.inserted_id}")
        return str(result.inserted_id)

    def update_gallery_image_face(self, face_id, id_maps):
        print(f"Updating gallery image face for face_id: {face_id}")
        try:
            face_id_obj = ObjectId(face_id) if isinstance(face_id, str) else face_id

            # Fetch the current document
            current_doc = self.gallery_image_faces_collection.find_one({"_id": face_id_obj})
            if not current_doc:
                print(f"No document found with face_id: {face_id}")
                return

            # Get the current idMaps
            current_id_maps = current_doc.get('idMaps', [])

            # Function to create a unique key for each idMap
            def get_id_map_key(id_map):
                return (
                    id_map.get('album_id'),
                    id_map.get('image_id'),
                    id_map.get('imageVersionId'),
                    ObjectId(id_map.get('ec2FaceId'))
                )

            # Create a dictionary of existing idMaps for easy lookup
            existing_id_maps = {get_id_map_key(id_map): id_map for id_map in current_id_maps}

            # Add new idMaps, updating existing ones if necessary
            for new_id_map in id_maps:
                key = get_id_map_key(new_id_map)
                if key in existing_id_maps:
                    # Update existing idMap with new data
                    existing_id_maps[key].update(new_id_map)
                else:
                    # Add new idMap
                    existing_id_maps[key] = new_id_map

            # Convert back to list
            updated_id_maps = list(existing_id_maps.values())

            # Perform the update
            result = self.gallery_image_faces_collection.update_one(
                {"_id": face_id_obj},
                {
                    "$set": {
                        "idMaps": updated_id_maps,
                        "updatedAt": int(time.time())
                    }
                }
            )

            print(f"Update result: {result.matched_count} document(s) matched, {result.modified_count} document(s) modified")

            if result.matched_count == 0:
                print(f"No document found with face_id: {face_id}")
            elif result.modified_count == 0:
                print("Document matched but not modified. This could mean the idMaps were already up to date.")

        except Exception as e:
            print(f"An error occurred while updating gallery image face: {str(e)}")
            raise

    
    def get_face_matching_images(self, image_ids, ec2_face_ids):
        print(f"Getting face matching images for {len(image_ids)} images and {len(ec2_face_ids)} ec2_face_ids")
        if len(image_ids) == 0:
            print("No image_ids provided")
            return []

        aggregation = [
            {
                "$match": {"_id": {"$in": [ObjectId(id) for id in image_ids]}, "activeVersionId": {"$ne": None}}
            },
            {
                "$unwind": {"path": "$versions", "preserveNullAndEmptyArrays": True}
            },
            {
                "$unwind": {"path": "$versions.faces", "preserveNullAndEmptyArrays": True}
            },
            {
                "$match": {"versions.faces.ec2FaceId": {"$in": ec2_face_ids}}
            },
            {
                "$project": {
                    "_id": 0,
                    "album_id": "$album_id",
                    "image_id": "$_id",
                    "imageVersionId": "$versions.versionId",
                    "ec2FaceId": "$versions.faces.ec2FaceId",
                    "boundingBox": "$versions.faces.boundingBox",
                    "optimizedImageS3Key": "$versions.s3_optimized.key",
                    "face_id": "$versions.faces.face_id",
                }
            }
        ]

        result = list(self.gallery_images_collection.aggregate(aggregation))
        print(f"Found {len(result)} matching images")
        return result
    
    def update_display_image(self, matching_face_id, display_image) -> None:
        try:
            # Prepare the new displayImage
            new_display_image = {
                "optimizedImageS3Key": display_image.get('optimizedImageS3Key'),
                "boundingBox": display_image.get('boundingBox'),
                "image_id":  ObjectId(display_image.get('image_id')) if isinstance(display_image.get('image_id'), str) else display_image.get('image_id')
            }

            matching_face_id = ObjectId(matching_face_id) if isinstance(matching_face_id, str) else matching_face_id

            # Update the document
            result = self.gallery_image_faces_collection.update_one(
                {"_id": matching_face_id},
                {"$set": {"displayImage": new_display_image}}
            )

            if result.modified_count > 0:
                print(f"Successfully updated displayImage for document {matching_face_id}")
            else:
                print(f"No changes made to displayImage for document {matching_face_id}")

        except Exception as e:
            print(f"Error updating displayImage for document {matching_face_id}: {str(e)}")


    def update_face_id_in_image_face(self, face_id, similar_faces):
        print(f"Updating face_id in image_face for face_id: {face_id}")
        print(f"similar_faces: {similar_faces}")
        print(f"face_id: {face_id}")
        try:
            face_id_obj = ObjectId(face_id) if isinstance(face_id, str) else face_id
            
            def update_single_face(image_id):
                try:
                    image_id_obj = ObjectId(image_id) if isinstance(image_id, str) else image_id
                    
                    result = self.gallery_faces_collection.update_one(
                        {"_id": image_id_obj},
                        {"$set": {"face_id": face_id_obj}}
                    )
                    
                    if result.modified_count > 0:
                        print(f"Successfully added face_id {face_id} to image {image_id}")
                        print(f"Successfully added face_id {face_id} to image {image_id}")
                    else:
                        print(f"No changes made for image {image_id}. Face_id might already be present.")
                        print(f"No changes made for image {image_id}. Face_id might already be present.")
                except Exception as e:
                    print(f"Error updating single face {image_id}: {str(e)}")
                    print(f"Error updating single face {image_id}: {str(e)}")

            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(update_single_face, similar_faces)
            print(f"Completed updating face_id in gallery_faces_collection for {len(similar_faces)} images")
            print(f"Completed updating face_id in gallery_faces_collection for {len(similar_faces)} images")
        except Exception as e:
            print(f"Error updating face_id in gallery_faces_collection: {str(e)}")
            print(f"Error updating face_id in gallery_faces_collection: {str(e)}")

print("mongodb_handler.py execution completed")
