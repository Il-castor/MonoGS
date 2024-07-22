import os
import time
from scipy.spatial.transform import Rotation as R
import glob
import csv
import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from pathlib import Path

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]
            
            
            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)

            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        # print(color_path)
        # print(type(color_path))
        image = cv2.imread(str(color_path), 0)
        image_r = cv2.imread(str(color_path_r), 0)
        depth = None
        if self.disorted:
            print("Eseguo rettifica")
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        # 47.90639384423901 is the baseline*fx from euroc 
        # questo valore l'ho preso da kitti04-12.yaml di orbslam3 
        depth = 379.8145 / (
            disparity
        )  ## Following ORB-SLAM3 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        
        # plt.imshow(depth, cmap='jet')
        # plt.colorbar()
        # plt.show()

        return image, depth, pose

class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose


class KittiParser:
    def __init__(self, input_folder, sequence, start_idx=0):
        self.input_folder = input_folder
        print(self.input_folder)
        self.sequence = sequence
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/image_2/*.png")
        )
        # print("color_path: ", self.color_paths)
        self.load_timestamps(
            f"{self.input_folder}/times.txt"
        )
        # print("timestamp ",self.timestamps)
        self.load_poses(
            f"{self.input_folder}/poses.txt"
        )
        # print("poses ", self.poses)
        self.load_calibration(
            f"{self.input_folder}/calib.txt"
        )
        # print("calib ", self.calib)
        self.n_img = len(self.color_paths)
        
    def load_timestamps(self, path):
        with open(path) as f:
            self.timestamps = [float(line.strip()) for line in f]
        self.timestamps = self.timestamps[self.start_idx:]
    
    
    # def load_poses(self, path):
    #     self.poses = []
    #     with open(path) as f:
    #         data = [list(map(float, line.split())) for line in f]
        
    #     #matrice in WORD to CAMERA reference system
    #     self.poses = []
    #     for i in range(self.start_idx, len(data)):
    #         T_w_c = np.eye(4)
    #         T_w_c[:3, :] = np.reshape(data[i], (3, 4))
    #         self.poses.append(T_w_c)
        
    #     self.frames = []
    #     #qua converto in CAMERA to WORD reference system
    #     for i, T_w_c in enumerate(self.poses):
    #         frame = {
    #             "file_path": self.color_paths[i],
    #             "timestamp": self.timestamps[i],
    #             "transform_matrix": np.linalg.inv(T_w_c).tolist(),
    #         }
    #         self.frames.append(frame)
    
    def load_poses(self, path):
        """Return the inverse pose matrices from the file."""
        self.poses = []
        with open(path, 'r') as f:
            for line in f: 
                values = list(map(float, line.strip().split()))
                pose = np.eye(4)
                pose[0:3, 0:4] = np.array(values).reshape(3, 4)
                inverted_pose = np.linalg.inv(pose)
                self.poses.append(inverted_pose)

        

    # load poses from file without any transformation
    # def load_poses(self, path):
    #     # Loading poses from file without any transformation
    #     # The poses are in world to camera reference system
    #     self.poses = []
    #     with open(path) as f:
    #         data = [list(map(float, line.split())) for line in f]
        
    #     self.poses = []
    #     for i in range(self.start_idx, len(data)):
    #         T_w_c = np.eye(4)
    #         T_w_c[:3, :] = np.reshape(data[i], (3, 4))
    #         self.poses.append(T_w_c)


    # vecchio codice 
    # def load_poses(self, path):
    #     self.poses = []
    #     with open(path) as f:
    #         data = [list(map(float, line.split())) for line in f]
        
    #     self.poses = []
    #     self.camera_to_world_matrices = []  # New list to store camera-to-world matrices
    #     for i in range(self.start_idx, len(data)):
    #         T_w_c = np.eye(4)
    #         T_w_c[:3, :] = np.reshape(data[i], (3, 4))
    #         T_c_w = np.linalg.inv(T_w_c)  # Invert the matrix to get camera-to-world
    #         self.poses.append(T_w_c)
    #         self.camera_to_world_matrices.append(T_c_w)  # Store the inverted matrix

    #     self.frames = []
    #     for i, T_w_c in enumerate(self.poses):
    #         frame = {
    #         "file_path": self.color_paths[i],
    #         "timestamp": self.timestamps[i],
    #         "transform_matrix": T_w_c.tolist(),  # This now stores world-to-camera
    #         "camera_to_world_matrix": self.camera_to_world_matrices[i].tolist()  # New field for camera-to-world
    #     }
    #     self.frames.append(frame)
            
    def load_calibration(self, path):
        with open(path) as f:
            lines = f.readlines()
        
        self.calib = {}
        for line in lines:
            key, value = line.split(":", 1)
            self.calib[key] = np.array(list(map(float, value.split())))
            
    def get_frames(self):
        return self.frames


class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KittiParser(dataset_path, 0)
        self.num_imgs = parser.n_img
        assert self.num_imgs > 0, "ERROR: no images found in dataset"
        self.color_paths = parser.color_paths
        self.poses = parser.poses
        self.timestamps = parser.timestamps

class KittiParserStereo:
    def __init__(self, sequence_dir):
        self.sequence_dir = Path(sequence_dir)
        self.image_0_dir = self.sequence_dir / 'image_0'
        self.image_1_dir = self.sequence_dir / 'image_1'
        self.times_file = self.sequence_dir / 'times.txt'
        print("sequenza: ", self.sequence_dir)
        self.poses_file = self.sequence_dir / 'poses.txt'
        print("Using poses file named", self.poses_file)
        self.timestamps = self._load_timestamps()
        self.poses = self.load_poses()
        self.color_paths = sorted(self.image_0_dir.glob('*.png'))
        self.color_paths_r = sorted(self.image_1_dir.glob('*.png'))
        self.n_img = len(self.color_paths)

    def _load_timestamps(self):
        with open(self.times_file, 'r') as file:
            timestamps = [float(line.strip()) for line in file]
        return timestamps

    # vecchio codice 
    # def _load_poses(self):
    #     poses = []
    #     if self.poses_file.exists():
    #         with open(self.poses_file, 'r') as file:
    #             for line in file:
    #                 pose = np.array(line.strip().split()).astype(np.float32).reshape(3, 4)  # Assuming poses are 3x4 matrices
    #                 poses.append(pose)
    #     else:
    #         print(f"Warning: Poses file {self.poses_file} not found.")
    #     return poses

    def load_poses(self):
        self.poses = []
        with open(self.poses_file, 'r') as file:
            for line in file:
                values = list(map(float, line.strip().split()))
                pose = np.eye(4)
                pose[0:3, 0:4] = np.array(values).reshape(3, 4)
                inverted_pose = np.linalg.inv(pose)
                self.poses.append(inverted_pose)

    def load_stereo_images(self, idx):
        """Load a pair of stereo images by index."""
        if idx < 0 or idx >= self.n_img:
            raise ValueError("Index out of bounds")
        left_image_path = self.color_paths[idx]
        right_image_path = self.color_paths_r[idx]
        # Here you would actually load the images using a library like OpenCV or PIL
        # For example: left_image = cv2.imread(str(left_image_path))
        return left_image_path, right_image_path

    def get_data(self):
        """Combine timestamps, poses, and image paths into a single structure."""
        data = []
        for idx in range(self.n_img):
            timestamp = self.timestamps[idx]
            pose = self.poses[idx] if idx < len(self.poses) else None
            left_image_path, right_image_path = self.load_stereo_images(idx)
            data_entry = {
                'timestamp': timestamp,
                'pose': pose,
                'left_image_path': left_image_path,
                'right_image_path': right_image_path,
            }
            data.append(data_entry)
        return data
    

class KittiStereoDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KittiParserStereo(dataset_path)

        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses



class BagParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        
        # Load image paths
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/left/*.png")
        )
        
        # Load timestamps
        with open(f"{self.input_folder}/timestamp_foto_left.txt", 'r') as f:
            self.timestamps = [int(line.strip()) for line in f]
        
        self.color_paths = self.color_paths[start_idx:]
        self.timestamps = self.timestamps[start_idx:]
        self.n_img = len(self.color_paths)
        
        # Load poses
        self.load_poses(f"{self.input_folder}/poses.txt")

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = self.timestamps[i] / 1e9  # Convert to seconds
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)
        
        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            data = [list(map(float, row)) for row in reader]
        
        data = np.array(data)
        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)
        
        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]  # Convert to w, x, y, z format
            
            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            
            T_w_c = T_w_i  # Assuming no additional transformation

            self.poses += [np.linalg.inv(T_w_c)]
            # print("dataset poses: ", self.poses)
            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }
            
            frames.append(frame)
        
        self.frames = frames

class BagDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = BagParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.poses = parser.poses



def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "kitti":
        print("sto usando file dataset.py")
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "kitti_stereo":
        return KittiStereoDataset(args, path, config)
    elif config["Dataset"]["type"] == "bag":
        return BagDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")