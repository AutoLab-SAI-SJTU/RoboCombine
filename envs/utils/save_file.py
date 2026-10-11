import numpy as np
from PIL import Image, ImageColor
import json
import os
import pickle
try:
    import open3d as o3d
    HAS_OPEN3D = True
except (ImportError, Exception):
    HAS_OPEN3D = False
    o3d = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_img(save_path, img_file):
    img = Image.fromarray(img_file)
    ensure_dir(save_path)
    img.save(save_path)

def save_json(save_path, json_file):
    ensure_dir(save_path)
    with open(save_path, 'w') as f:
        json.dump(json_file, f, indent=4)

def save_pkl(save_path, dic_file):
    ensure_dir(save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(dic_file, f)

def save_pcd(save_path, pcd_arr, color=False):
    if not HAS_OPEN3D:
        print(f'Warning: open3d not available, skipping point cloud save: {save_path}')
        return
    ensure_dir(save_path)
    point_cloud = o3d.geometry.PointCloud()
    point_arr = pcd_arr[:, :3]
    point_cloud.points = o3d.utility.Vector3dVector(point_arr)
    if color:
        colors_arr = pcd_arr[:, -3:]
        point_cloud.colors = o3d.utility.Vector3dVector(colors_arr)
    o3d.io.write_point_cloud(save_path, point_cloud)
