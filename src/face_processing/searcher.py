import faiss
import torch
import numpy as np
import psutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json



class GPUMultiWorkerGrouper:
    def __init__(self):
        # Check if CUDA is available
        print(f"\nCUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available, falling back to CPU!")
            
        self.device = torch.device('cuda')

        self.n_initial_workers = psutil.cpu_count()
        self.n_merge_workers   = 2

        self.streams = [
            torch.cuda.Stream() 
            for _ in range(self.n_initial_workers + self.n_merge_workers)
        ]
        self.gpu_resources = [
            faiss.StandardGpuResources()
            for _ in range(max(self.n_initial_workers, self.n_merge_workers))
        ]

        self.base_threshold  = 0.70
        self.side_threshold  = 0.66
        self.merge_threshold = 0.71

        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        self.working_memory = int(gpu_mem * 0.9)

        for res in self.gpu_resources:
            res.setTempMemory(self.working_memory // len(self.gpu_resources))

        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)

        # Print resource allocation info
        print(f"Number of CPU workers: {self.n_initial_workers}")
        print(f"Working memory allocated: {self.working_memory / (1024**3):.2f} GB")
        print("Initialization complete!\n")

    # PHASE 1
    def calculate_batch_size(self, total_faces: int) -> Tuple[int, int]:
        gpu_mem = self.working_memory / (1024**3)
        embedding_size = 512 * 4 / (1024**2)
        
        if total_faces < 10000:
            batch_size = total_faces
            sub_batch = 500
        elif total_faces < 100000:
            batch_size = min(50000, total_faces // self.n_initial_workers)
            sub_batch = 1000
        else:
            available_mem = gpu_mem * 0.8
            max_embeddings = int(available_mem * 1024 / embedding_size)
            batch_size = min(max_embeddings, total_faces // self.n_initial_workers)
            sub_batch = min(2000, batch_size // 10)
        
        return batch_size, sub_batch

    # PHASE 2
    def process_batch(
        self,
        embeddings: torch.Tensor,
        faces: list,
        start_idx: int,
        end_idx: int,
        worker_id: int
    ) -> List[List[Dict[str, Any]]]:
        with torch.cuda.stream(self.streams[worker_id]):
            print(f"Worker {worker_id}: Processing {end_idx - start_idx} faces")
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_faces      = faces[start_idx:end_idx]

            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = True
            index = faiss.GpuIndexFlatIP(self.gpu_resources[worker_id], embeddings.shape[1], cfg)
            index.add(batch_embeddings.cpu().numpy())

            groups = []
            processed = set()

            for i in range(len(batch_faces)):
                if i in processed:
                    continue
                D, I = index.search(
                    batch_embeddings[i:i+1].cpu().numpy(), 
                    min(100, len(batch_embeddings))
                )
                group = self._form_group(
                    index, D[0], I[0], 
                    batch_embeddings, batch_faces, 
                    i, processed, start_idx
                )
                if group:
                    groups.append(group)

            torch.cuda.current_stream().synchronize()
            return groups

    def _form_group(
        self,
        index: faiss.GpuIndexFlatIP,
        distances: np.ndarray,
        indices: np.ndarray,
        embeddings: torch.Tensor,
        faces: List[Dict[str, Any]],
        current_idx: int,
        processed: set,
        offset: int
    ) -> List[Dict[str, Any]]:
        group = []
        face1 = faces[current_idx]
        group.append({
            'idx': current_idx + offset,
            'data': face1,
            'similarity': 1.0
        })
        processed.add(current_idx)

        for dist, idx in zip(distances, indices):
            if idx in processed or idx == current_idx:
                continue
            face2 = faces[idx]
            sim_score = (1 + dist) / 2
            threshold = self._get_threshold(face1, face2)

            if sim_score >= threshold:
                # Reverse check
                rev_D, _ = index.search(embeddings[idx:idx+1].cpu().numpy(), 1)
                rev_sim = (1 + rev_D[0][0]) / 2
                if rev_sim >= threshold:
                    group.append({
                        'idx': idx + offset,
                        'data': face2,
                        'similarity': sim_score
                    })
                    processed.add(idx)

        return group

    def _get_threshold(self, face1: Dict[str, Any], face2: Dict[str, Any]) -> float:
        if (
            face1.orientation == "Front" and face1.orientation_percentage >= 75 and
            face2.orientation == "Front" and face2.orientation_percentage >= 75
        ):
            return self.base_threshold

        is_side = (
            (face1.orientation in ['Side Left', 'Side Right', 'Down'] and face1.orientation_percentage >= 75) or
            (face2.orientation in ['Side Left', 'Side Right', 'Down'] and face2.orientation_percentage >= 75)
        )
        return self.side_threshold if is_side else self.base_threshold

    # PHASE 3
    def merge_groups(
        self,
        all_groups: List[List[Dict[str, Any]]],
        embeddings: torch.Tensor,
        worker_id: int
    ) -> List[Dict[str, Any]]:
        print(f"Merge worker {worker_id}: Starting group merge process")
        with torch.cuda.stream(self.streams[self.n_initial_workers + worker_id]):
            if not all_groups:
                return []

            # 1) best group to the front
            best_center_score = -1
            best_center_group_idx = 0
            for g_idx, group in enumerate(all_groups):
                c_idx = self._get_best_center(group)
                c_face = group[c_idx]['data']
                score = c_face.orientation_percentage
                if c_face.orientation == "Front":
                    score *= 1.2
                if score > best_center_score:
                    best_center_score = score
                    best_center_group_idx = g_idx

            best_group = all_groups[best_center_group_idx]
            all_groups = [best_group] + all_groups[:best_center_group_idx] + all_groups[best_center_group_idx+1:]

            # 2) gather center faces
            center_faces = []
            for group in all_groups:
                c_idx = self._get_best_center(group)
                center_faces.append({
                    'embedding': embeddings[group[c_idx]['idx']],
                    'orientation': group[c_idx]['data'].orientation,
                    'percentage': group[c_idx]['data'].orientation_percentage,
                    'group': group
                })

            # 3) index center embeddings
            center_embeddings = torch.stack([c['embedding'] for c in center_faces])
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = True
            index = faiss.GpuIndexFlatIP(self.gpu_resources[worker_id], center_embeddings.shape[1], cfg)
            index.add(center_embeddings.cpu().numpy())

            final_groups = []
            processed = set()

            # 4) find merges
            for i in range(len(center_faces)):
                if i in processed:
                    continue
                D, I = index.search(center_embeddings[i:i+1].cpu().numpy(), min(100, len(center_embeddings)))
                merged_group = self._merge_groups(D[0], I[0], center_faces, i, processed)
                if merged_group:
                    final_groups.append(merged_group)

            torch.cuda.current_stream().synchronize()
            print(f"Merging complete. Found {len(final_groups)} final groups")
            return self._format_groups_for_json(final_groups)

    def _merge_groups(self, distances, indices, center_faces, current_idx, processed):
        merged_group = []
        face1 = center_faces[current_idx]

        for dist, idx in zip(distances, indices):
            if idx in processed:
                continue
            face2 = center_faces[idx]
            sim_score = (1 + dist) / 2
            threshold = self.merge_threshold

            is_face1_front = (face1['orientation'] == "Front")
            is_face2_front = (face2['orientation'] == "Front")
            is_face1_side  = (face1['orientation'] in ['Side Left', 'Side Right', 'Down'])
            is_face2_side  = (face2['orientation'] in ['Side Left', 'Side Right', 'Down'])
            face1_high_conf = face1['percentage'] >= 75
            face2_high_conf = face2['percentage'] >= 75

            # Adjust threshold for front-side or side-side combos
            if (is_face1_front and is_face2_side and face2_high_conf) or \
               (is_face2_front and is_face1_side and face1_high_conf):
                threshold = 0.66
            elif is_face1_side and is_face2_side and face1_high_conf and face2_high_conf:
                threshold = 0.70
            elif is_face1_side and is_face2_side and (not face1_high_conf or not face2_high_conf):
                threshold = 0.695

            if sim_score >= threshold:
                merged_group.extend(face2['group'])
                processed.add(idx)

        return merged_group if merged_group else None

    def _get_best_center(self, group: List[Dict[str, Any]]) -> int:
        best_score = -1
        best_idx   = 0
        for idx, face_rec in enumerate(group):
            f = face_rec['data']
            score = f.orientation_percentage
            if f.orientation == "Front":
                score *= 1.2
            if f.face_confidence >= 0.55:
                score *= 1.1
            if score > best_score:
                best_score = score
                best_idx   = idx
        return best_idx

    def _format_groups_for_json(self, groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        results = []
        for i, group in enumerate(groups, start=1):
            if not group:  # Skip empty groups
                continue
            
            # Get the first face as center face
            center_face_entry = group[0]
            center_face = center_face_entry['data']
            
            # Create members list from remaining faces
            members = []
            for face_entry in group[1:]:  # Start from second face
                members.append({
                    "face": {
                        **face_entry['data'].to_dict(),
                        "similarity": float(face_entry["similarity"]),
                        "embedding": []
                    }
                })
            
            results.append({
                "group_id": i,
                "center_face": {
                    **center_face.to_dict(),
                    "similarity": 1.0,
                    "embedding": []
                },
                "members": members
            })
        return results


    # Single convenience method
    def group_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"\nStarting face grouping for {len(faces)} faces...")
        
        if not faces:
            return []

        print("Loading embeddings to device...")
        embeddings = torch.stack([torch.from_numpy(f.embedding) for f in faces]).to(self.device)

        batch_size, search_size = self.calculate_batch_size(len(faces))
        print(f"Batch size: {batch_size}, Search size: {search_size}")

        print("\nPhase 1: Initial grouping...")
        all_groups = []
        start = 0
        worker_id = 0
        while start < len(faces):
            end = min(start + batch_size, len(faces))
            print(f"Processing batch {worker_id + 1}: faces {start} to {end}")
            batch_groups = self.process_batch(embeddings, faces, start, end, worker_id)
            all_groups.extend(batch_groups)
            print(f"Found {len(batch_groups)} groups in this batch")
            start = end
            worker_id = (worker_id + 1) % self.n_initial_workers

        print(f"\nPhase 2: Merging {len(all_groups)} groups...")
        final_result = self.merge_groups(all_groups, embeddings, worker_id=0)
        print(f"Final number of groups: {len(final_result)}")
        
        return final_result
