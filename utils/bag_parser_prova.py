# import os
# import numpy as np

# class DatasetLoader:
#     def __init__(self, input_folder):
#         self.input_folder = input_folder
#         self.image_timestamps_file = f"{self.input_folder}/timestamp_foto_left.txt"
#         self.poses_file = f"{self.input_folder}/poses.txt"
#         self.images_folder = f"{self.input_folder}/left/"
#         self.image_timestamps = []
#         self.pose_data = []
#         self.pose_timestamps = []
#         self.associations = []
#         self.data = []

#     def load_image_timestamps(self):
#         with open(self.image_timestamps_file, 'r') as f:
#             self.image_timestamps = [int(line.strip()) for line in f]

#     def load_pose_data(self):
#         with open(self.poses_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 timestamp = float(parts[0])
#                 pose = [float(p) for p in parts[1:]]
#                 self.pose_data.append((timestamp, pose))
#         # Convert pose timestamps to integer format (nanoseconds)
#         self.pose_timestamps = [int(ts * 1e9) for ts, _ in self.pose_data]

#     def find_closest_pose_timestamp(self, image_ts):
#         pose_ts_array = np.array(self.pose_timestamps)
#         index = np.argmin(np.abs(pose_ts_array - image_ts))
#         return index

#     def associate_images_with_poses(self):
#         for image_ts in self.image_timestamps:
#             closest_index = self.find_closest_pose_timestamp(image_ts)
#             closest_pose = self.pose_data[closest_index]
#             image_filename = os.path.join(self.images_folder, f"{image_ts}.png")
#             self.associations.append((image_filename, closest_pose))

#     def save_associations(self):
#         association_list = []
        
#         for image_filename, (pose_ts, pose) in self.associations:
#             pose_str = ' '.join(map(str, pose))
#             association_list.append(f"{image_filename} {pose_ts} {pose_str}")
#         return association_list

#     def load_and_associate(self):
#         self.load_image_timestamps()
#         self.load_pose_data()
#         self.associate_images_with_poses()

#         self.data = self.save_associations()
#         self.n_imgs = len(self.associations)
#         self.color_path = self.associations

# # Usage
# input_path = '/media/castor/T9/tirocinio/zed2'
# dataset_loader = DatasetLoader(input_path)

# dataset_loader.load_and_associate()
# # print("COCCO ", dataset_loader.data)
# print("COCCO", dataset_loader.n_imgs)
# print("COCCO ", dataset_loader.color_path)

import os
import numpy as np
import trimesh

class BagDataset:
    def __init__(self, image_timestamps_file, poses_file, images_folder, frame_rate=32):
        self.image_timestamps_file = image_timestamps_file
        self.poses_file = poses_file
        self.images_folder = images_folder
        self.frame_rate = frame_rate
        self.load_poses()
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_pose, max_dt=3.00):
        associations = []
        for i, t in enumerate(tstamp_image):
            k = np.argmin(np.abs(tstamp_pose - t))
            if np.abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, k))
        return associations

    def load_poses(self):
        # Load image timestamps
        image_timestamps = self.parse_list(self.image_timestamps_file)
        if image_timestamps.ndim == 1:
            tstamp_image = image_timestamps.astype(np.float64)
        else:
            tstamp_image = image_timestamps[:, 0].astype(np.float64)
        
        # Load pose data
        pose_data = self.parse_list(self.poses_file, skiprows=1)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        
        # Associate frames
        associations = self.associate_frames(tstamp_image, tstamp_pose)

        # Filter based on frame rate
        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / self.frame_rate:
                indices.append(i)

        # Initialize lists for storing paths and poses
        self.color_paths, self.poses, self.frames = [], [], []
        print("CIAO ", len(associations))
        for ix in indices:
            i, k = associations[ix]
            self.color_paths.append(os.path.join(self.images_folder, f"{int(tstamp_image[i])}.png"))

            quat = pose_vecs[k][3:]
            trans = pose_vecs[k][:3]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses.append(np.linalg.inv(T))

            frame = {
                "file_path": str(os.path.join(self.images_folder, f"{int(tstamp_image[i])}.png")),
                "transform_matrix": np.linalg.inv(T).tolist(),
            }
            self.frames.append(frame)

    def save_associations(self, output_file):
        with open(output_file, 'w') as f:
            for frame in self.frames:
                file_path = frame["file_path"]
                transform_matrix = frame["transform_matrix"]
                transform_str = ' '.join(map(str, np.array(transform_matrix).flatten()))
                f.write(f"{file_path} {transform_str}\n")

# Usage
bag_dataset = BagDataset(
    image_timestamps_file='/media/castor/T9/tirocinio/zed2/timestamp_foto_left.txt',
    poses_file='/media/castor/T9/tirocinio/zed2/poses.txt',
    images_folder='/media/castor/T9/tirocinio/zed2/left'
)

bag_dataset.save_associations('image_pose_associations.txt')
print("DONE")