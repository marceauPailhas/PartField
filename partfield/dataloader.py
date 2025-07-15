import torch
import boto3
import json
from os import path as osp
# from botocore.config import Config
# from botocore.exceptions import ClientError
import h5py
import io
import numpy as np
import skimage
import trimesh
import os
from scipy.spatial import KDTree
import gc
from plyfile import PlyData

## For remeshing
import mesh2sdf
import tetgen
import vtk
import math
import tempfile

### For mesh processing
import pymeshlab

from partfield.utils import *

#########################
## To handle quad inputs
#########################
def quad_to_triangle_mesh(F):
    """
    Converts a quad-dominant mesh into a pure triangle mesh by splitting quads into two triangles.

    Parameters:
        quad_mesh (trimesh.Trimesh): Input mesh with quad faces.

    Returns:
        trimesh.Trimesh: A new mesh with only triangle faces.
    """
    faces = F

    ### If already a triangle mesh -- skip
    if len(faces[0]) == 3:
        return F

    new_faces = []

    for face in faces:
        if len(face) == 4:  # Quad face
            # Split into two triangles
            new_faces.append([face[0], face[1], face[2]])  # Triangle 1
            new_faces.append([face[0], face[2], face[3]])  # Triangle 2
        else:
            print(f"Warning: Skipping non-triangle/non-quad face {face}")

    new_faces = np.array(new_faces)

    return new_faces
#########################

class Demo_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.data_path = cfg.dataset.data_path
        self.is_pc = cfg.is_pc

        all_files = os.listdir(self.data_path)

        selected = []
        for f in all_files:
            if ".ply" in f and self.is_pc:
                selected.append(f)
            elif (".obj" in f or ".glb" in f or ".off" in f) and not self.is_pc:
                selected.append(f)

        self.data_list = selected
        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)

    def load_ply_to_numpy(self, filename):
        """
        Load a PLY file and extract the point cloud as a (N, 3) NumPy array.

        Parameters:
            filename (str): Path to the PLY file.

        Returns:
            numpy.ndarray: Point cloud array of shape (N, 3).
        """
        ply_data = PlyData.read(filename)

        # Extract vertex data
        vertex_data = ply_data["vertex"]
        
        # Convert to NumPy array (x, y, z)
        points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T

        return points

    def get_model(self, ply_file):

        uid = ply_file.split(".")[-2].replace("/", "_")

        ####
        if self.is_pc:
            ply_file_read = os.path.join(self.data_path, ply_file)
            pc = self.load_ply_to_numpy(ply_file_read)

            bbmin = pc.min(0)
            bbmax = pc.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / (bbmax - bbmin).max()
            pc = (pc - center) * scale

        else:
            obj_path = os.path.join(self.data_path, ply_file)
            mesh = load_mesh_util(obj_path)
            vertices = mesh.vertices
            faces = mesh.faces

            bbmin = vertices.min(0)
            bbmax = vertices.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / (bbmax - bbmin).max()
            vertices = (vertices - center) * scale
            mesh.vertices = vertices

            ### Make sure it is a triangle mesh -- just convert the quad
            mesh.faces = quad_to_triangle_mesh(faces)

            print("before preprocessing...")
            print(mesh.vertices.shape)
            print(mesh.faces.shape)
            print()

            ### Pre-process mesh
            if self.preprocess_mesh:
                # Create a PyMeshLab mesh directly from vertices and faces
                ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

                # Create a MeshSet and add your mesh
                ms = pymeshlab.MeshSet()
                ms.add_mesh(ml_mesh, "from_trimesh")

                # Apply filters
                ms.apply_filter('meshing_remove_duplicate_faces')
                ms.apply_filter('meshing_remove_duplicate_vertices')
                percentageMerge = pymeshlab.PercentageValue(0.5)
                ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
                ms.apply_filter('meshing_remove_unreferenced_vertices')

                # Save or extract mesh
                processed = ms.current_mesh()
                mesh.vertices = processed.vertex_matrix()
                mesh.faces = processed.face_matrix()               

                print("after preprocessing...")
                print(mesh.vertices.shape)
                print(mesh.faces.shape)

            ### Save input
            save_dir = f"exp_results/{self.result_name}"
            os.makedirs(save_dir, exist_ok=True)
            view_id = 0            
            mesh.export(f'{save_dir}/input_{uid}_{view_id}.ply')                


            pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 

        result = {
                    'uid': uid
                }

        result['pc'] = torch.tensor(pc, dtype=torch.float32)

        if not self.is_pc:
            result['vertices'] = torch.tensor(mesh.vertices, dtype=torch.float32)
            result['faces'] = torch.tensor(mesh.faces, dtype=torch.float32)

        return result

    def __getitem__(self, index):
        
        gc.collect()

        return self.get_model(self.data_list[index])

##############

###############################
class Demo_Remesh_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()

        self.data_path = cfg.dataset.data_path

        all_files = os.listdir(self.data_path)

        selected = []
        for f in all_files:
            if (".obj" in f or ".glb" in f):
                selected.append(f)

        self.data_list = selected
        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)


    def get_model(self, ply_file):

        uid = ply_file.split(".")[-2]

        ####
        obj_path = os.path.join(self.data_path, ply_file)
        mesh =  load_mesh_util(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces

        bbmin = vertices.min(0)
        bbmax = vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * 0.9 / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale
        mesh.vertices = vertices

        ### Pre-process mesh
        if self.preprocess_mesh:
            # Create a PyMeshLab mesh directly from vertices and faces
            ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)

            # Create a MeshSet and add your mesh
            ms = pymeshlab.MeshSet()
            ms.add_mesh(ml_mesh, "from_trimesh")

            # Apply filters
            ms.apply_filter('meshing_remove_duplicate_faces')
            ms.apply_filter('meshing_remove_duplicate_vertices')
            percentageMerge = pymeshlab.PercentageValue(0.5)
            ms.apply_filter('meshing_merge_close_vertices', threshold=percentageMerge)
            ms.apply_filter('meshing_remove_unreferenced_vertices')


            # Save or extract mesh
            processed = ms.current_mesh()
            mesh.vertices = processed.vertex_matrix()
            mesh.faces = processed.face_matrix()               

            print("after preprocessing...")
            print(mesh.vertices.shape)
            print(mesh.faces.shape)

        ### Save input
        save_dir = f"exp_results/{self.result_name}"
        os.makedirs(save_dir, exist_ok=True)
        view_id = 0            
        mesh.export(f'{save_dir}/input_{uid}_{view_id}.ply')   

        try:
            ###### Remesh ######
            size= 256
            level = 2 / size

            sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
            # NOTE: the negative value is not reliable if the mesh is not watertight
            udf = np.abs(sdf)
            vertices, faces, _, _ = skimage.measure.marching_cubes(udf, level)

            #### Only use SDF mesh ###
            # new_mesh = trimesh.Trimesh(vertices, faces)
            ##########################

            #### Make tet #####
            components = trimesh.Trimesh(vertices, faces).split(only_watertight=False)
            new_mesh = [] #trimesh.Trimesh()
            if len(components) > 100000:
                raise NotImplementedError
            for i, c in enumerate(components):
                c.fix_normals()
                new_mesh.append(c) #trimesh.util.concatenate(new_mesh, c)
            new_mesh = trimesh.util.concatenate(new_mesh)

            # generate tet mesh
            tet = tetgen.TetGen(new_mesh.vertices, new_mesh.faces)
            tet.tetrahedralize(plc=True, nobisect=1., quality=True, fixedvolume=True, maxvolume=math.sqrt(2) / 12 * (2 / size) ** 3)
            tmp_vtk = tempfile.NamedTemporaryFile(suffix='.vtk', delete=True)
            tet.grid.save(tmp_vtk.name)

            # extract surface mesh from tet mesh
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(tmp_vtk.name)
            reader.Update()
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(reader.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()
            writer = vtk.vtkOBJWriter()
            tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=True)
            writer.SetFileName(tmp_obj.name)
            writer.SetInputData(polydata)
            writer.Update()
            new_mesh =  load_mesh_util(tmp_obj.name)
            ##########################

            new_mesh.vertices = new_mesh.vertices * (2.0 / size) - 1.0  # normalize it to [-1, 1]

            mesh = new_mesh
        ####################

        except:
            print("Error in tet.")
            mesh = mesh 

        pc, _ = trimesh.sample.sample_surface(mesh, self.pc_num_pts) 

        result = {
                    'uid': uid
                }

        result['pc'] = torch.tensor(pc, dtype=torch.float32)
        result['vertices'] = torch.tensor(mesh.vertices, dtype=torch.float32)
        result['faces'] = torch.tensor(mesh.faces, dtype=torch.float32)

        return result

    def __getitem__(self, index):
        
        gc.collect()

        return self.get_model(self.data_list[index])


class Correspondence_Demo_Dataset(Demo_Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.data_path = cfg.dataset.data_path
        self.is_pc = cfg.is_pc

        self.data_list = cfg.dataset.all_files

        self.pc_num_pts = 100000

        self.preprocess_mesh = cfg.preprocess_mesh
        self.result_name = cfg.result_name

        print("val dataset len:", len(self.data_list))


class Training_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.data_path = cfg.dataset.data_path
        self.pc_num_pts = cfg.dataset.pc_num_pts if hasattr(cfg.dataset, 'pc_num_pts') else 100000
        self.n_sdf_samples = cfg.dataset.n_sdf_samples if hasattr(cfg.dataset, 'n_sdf_samples') else 100000
        
        # Load training data list
        if hasattr(cfg.dataset, 'train_list'):
            with open(cfg.dataset.train_list, 'r') as f:
                self.data_list = [line.strip() for line in f]
        else:
            # Default: load all files from training directory
            train_dir = os.path.join(self.data_path, 'train')
            self.data_list = []
            for file in os.listdir(train_dir):
                if file.endswith(('.obj', '.ply', '.off', '.h5')):
                    self.data_list.append(os.path.join(train_dir, file))
        
        print(f"Training dataset size: {len(self.data_list)}")
    
    def __len__(self):
        return len(self.data_list)
    
    def load_shape_data(self, file_path):
        """Load shape data from various formats"""
        if file_path.endswith('.h5'):
            # Load from HDF5 file (typical for processed training data)
            with h5py.File(file_path, 'r') as f:
                pc = f['pc'][:]
                sdf_points = f['sdf_points'][:] if 'sdf_points' in f else None
                sdf_values = f['sdf_values'][:] if 'sdf_values' in f else None
                part_labels = f['part_labels'][:] if 'part_labels' in f else None
                return pc, sdf_points, sdf_values, part_labels
        else:
            # Load from mesh file
            mesh = load_mesh_util(file_path)
            
            # Normalize mesh
            vertices = mesh.vertices
            bbmin = vertices.min(0)
            bbmax = vertices.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / (bbmax - bbmin).max()
            vertices = (vertices - center) * scale
            mesh.vertices = vertices
            
            # Sample point cloud
            pc, face_indices = trimesh.sample.sample_surface(mesh, self.pc_num_pts)
            
            # Generate SDF samples
            sdf_points, sdf_values = self.generate_sdf_samples(mesh)
            
            return pc, sdf_points, sdf_values, None
    
    def generate_sdf_samples(self, mesh):
        """Generate SDF samples for training"""
        # Sample points near surface
        surface_points, _ = trimesh.sample.sample_surface(mesh, self.n_sdf_samples // 2)
        
        # Add noise to surface points
        noise = np.random.normal(0, 0.01, surface_points.shape)
        near_surface_points = surface_points + noise
        
        # Sample random points in volume
        bbox_min = mesh.vertices.min(0) - 0.1
        bbox_max = mesh.vertices.max(0) + 0.1
        random_points = np.random.uniform(bbox_min, bbox_max, (self.n_sdf_samples // 2, 3))
        
        # Combine all sample points
        sample_points = np.concatenate([near_surface_points, random_points], axis=0)
        
        # Compute SDF values (simplified - in practice you'd use a proper SDF computation)
        # This is a placeholder - you should implement proper SDF computation
        sdf_values = np.zeros(len(sample_points))
        for i, point in enumerate(sample_points):
            # Simple distance to mesh approximation
            distances = np.linalg.norm(mesh.vertices - point, axis=1)
            sdf_values[i] = distances.min()
        
        return sample_points, sdf_values
    
    def sample_point_pairs(self, pc, part_labels=None):
        """Sample point pairs for contrastive learning"""
        n_points = len(pc)
        
        if part_labels is not None:
            # Sample based on part labels
            unique_parts = np.unique(part_labels)
            
            # Sample points from same part
            same_part_indices = []
            diff_part_indices = []
            
            for part_id in unique_parts:
                part_points = np.where(part_labels == part_id)[0]
                if len(part_points) > 1:
                    # Sample pairs from same part
                    pair_indices = np.random.choice(part_points, size=min(1000, len(part_points)), replace=True)
                    same_part_indices.extend(pair_indices)
                    
                    # Sample from different parts
                    other_parts = np.where(part_labels != part_id)[0]
                    if len(other_parts) > 0:
                        diff_indices = np.random.choice(other_parts, size=min(1000, len(other_parts)), replace=True)
                        diff_part_indices.extend(diff_indices)
            
            pc_same_part = pc[same_part_indices] if same_part_indices else pc[:1000]
            pc_diff_part = pc[diff_part_indices] if diff_part_indices else pc[1000:2000]
        else:
            # Random sampling without part information
            indices = np.random.choice(n_points, size=min(2000, n_points), replace=False)
            pc_same_part = pc[indices[:1000]]
            pc_diff_part = pc[indices[1000:2000]]
        
        return pc_same_part, pc_diff_part
    
    def sample_triplets(self, pc, part_labels=None):
        """Sample triplets for triplet loss"""
        n_points = len(pc)
        
        if part_labels is not None:
            unique_parts = np.unique(part_labels)
            
            anchors = []
            positives = []
            negatives = []
            
            for part_id in unique_parts:
                part_points = np.where(part_labels == part_id)[0]
                other_points = np.where(part_labels != part_id)[0]
                
                if len(part_points) > 1 and len(other_points) > 0:
                    # Sample anchor from this part
                    anchor_idx = np.random.choice(part_points)
                    anchors.append(anchor_idx)
                    
                    # Sample positive from same part
                    pos_candidates = part_points[part_points != anchor_idx]
                    pos_idx = np.random.choice(pos_candidates)
                    positives.append(pos_idx)
                    
                    # Sample negative from different part
                    neg_idx = np.random.choice(other_points)
                    negatives.append(neg_idx)
            
            if anchors:
                return pc[anchors], pc[positives], pc[negatives]
        
        # Fallback to random sampling
        indices = np.random.choice(n_points, size=min(3000, n_points), replace=False)
        return pc[indices[:1000]], pc[indices[1000:2000]], pc[indices[2000:3000]]
    
    def __getitem__(self, index):
        file_path = self.data_list[index]
        
        try:
            pc, sdf_points, sdf_values, part_labels = self.load_shape_data(file_path)
            
            # Sample point pairs for contrastive learning
            pc_same_part, pc_diff_part = self.sample_point_pairs(pc, part_labels)
            
            # Sample triplets for triplet loss
            anchor_points, pos_points, neg_points = self.sample_triplets(pc, part_labels)
            
            result = {
                'pc': torch.tensor(pc, dtype=torch.float32),
                'pc_same_part': torch.tensor(pc_same_part, dtype=torch.float32),
                'pc_diff_part': torch.tensor(pc_diff_part, dtype=torch.float32),
                'anchor_points': torch.tensor(anchor_points, dtype=torch.float32),
                'pos_points': torch.tensor(pos_points, dtype=torch.float32),
                'neg_points': torch.tensor(neg_points, dtype=torch.float32),
            }
            
            if sdf_points is not None and sdf_values is not None:
                result['query_points'] = torch.tensor(sdf_points, dtype=torch.float32)
                result['sdf_gt'] = torch.tensor(sdf_values, dtype=torch.float32)
            
            if part_labels is not None:
                result['part_labels'] = torch.tensor(part_labels, dtype=torch.long)
            
            return result
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy batch
            dummy_pc = np.random.randn(self.pc_num_pts, 3).astype(np.float32)
            return {
                'pc': torch.tensor(dummy_pc, dtype=torch.float32),
                'pc_same_part': torch.tensor(dummy_pc[:1000], dtype=torch.float32),
                'pc_diff_part': torch.tensor(dummy_pc[1000:2000], dtype=torch.float32),
                'anchor_points': torch.tensor(dummy_pc[2000:3000], dtype=torch.float32),
                'pos_points': torch.tensor(dummy_pc[3000:4000], dtype=torch.float32),
                'neg_points': torch.tensor(dummy_pc[4000:5000], dtype=torch.float32),
            }


class Validation_Dataset(Training_Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Load validation data list
        if hasattr(cfg.dataset, 'val_list'):
            with open(cfg.dataset.val_list, 'r') as f:
                self.data_list = [line.strip() for line in f]
        else:
            # Default: load all files from validation directory
            val_dir = os.path.join(self.data_path, 'val')
            self.data_list = []
            for file in os.listdir(val_dir):
                if file.endswith(('.obj', '.ply', '.off', '.h5')):
                    self.data_list.append(os.path.join(val_dir, file))
        
        print(f"Validation dataset size: {len(self.data_list)}")
