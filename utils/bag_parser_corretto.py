import os
import numpy as np

class DatasetLoader:
    def __init__(self, image_timestamps_file, poses_file, images_folder):
        self.image_timestamps_file = image_timestamps_file
        self.poses_file = poses_file
        self.images_folder = images_folder
        self.image_timestamps = []
        self.pose_data = []
        self.pose_timestamps = []
        self.associations = []

    def load_image_timestamps(self):
        with open(self.image_timestamps_file, 'r') as f:
            self.image_timestamps = [int(line.strip()) for line in f]

    def load_pose_data(self):
        with open(self.poses_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                timestamp = float(parts[0])
                pose = [float(p) for p in parts[1:]]
                self.pose_data.append((timestamp, pose))
        # Convert pose timestamps to integer format (nanoseconds)
        self.pose_timestamps = [int(ts * 1e9) for ts, _ in self.pose_data]

    def find_closest_pose_timestamp(self, image_ts):
        pose_ts_array = np.array(self.pose_timestamps)
        index = np.argmin(np.abs(pose_ts_array - image_ts))
        return index

    def associate_images_with_poses(self):
        for image_ts in self.image_timestamps:
            closest_index = self.find_closest_pose_timestamp(image_ts)
            closest_pose = self.pose_data[closest_index]
            image_filename = os.path.join(self.images_folder, f"{image_ts}.png")
            self.associations.append((image_filename, closest_pose))

    def save_associations(self, output_file):
        with open(output_file, 'w') as f:
            for image_filename, (pose_ts, pose) in self.associations:
                pose_str = ' '.join(map(str, pose))
                f.write(f"{image_filename} {pose_ts} {pose_str}\n")

    def load_and_associate(self, output_file):
        self.load_image_timestamps()
        self.load_pose_data()
        self.associate_images_with_poses()
        self.save_associations(output_file)

# Usage
dataset_loader = DatasetLoader(
    image_timestamps_file='/media/castor/T9/tirocinio/zed2/timestamp_foto_left.txt',
    poses_file='/media/castor/T9/tirocinio/zed2/poses.txt.bk',
    images_folder='/media/castor/T9/tirocinio/zed2/left'
)

dataset_loader.load_and_associate('image_pose_associations.txt')
