import numpy as np
import pickle
import zipfile
from tqdm import tqdm

# CONFIGURATION
INPUT_PATH = '/home/motion/Downloads/h36m_sh_conf_cam_source_final.pkl.zip'
OUTPUT_PATH = 'h36m_train_3d_27frames.npy'
RECEPTIVE_FIELD = 27  # Size of the sliding window
STRIDE = 1            # How much we move the window (1 = max overlap)
IMG_RES = 1000.0      # Human3.6M Standard Crop Resolution


# Normalize X and Y
def normalize_pose(pose_arr):
    norm_pose = (pose_arr / IMG_RES) * 2 -1
    return norm_pose

# Standard Sliding Window Logic
def get_chunks(data_list, window_size, stride):

    if len(data_list) < window_size:
        return []

    chunks = []

    for i in range(0, len(data_list) - window_size + 1, stride):
        chunk = data_list[i : i + window_size]
        chunks.append(chunk)
    return chunks


def process_data():
    print(f"📂 loading raw data from {INPUT_PATH}...")

    # 1. Load Pickle from Zip
    with zipfile.ZipFile(INPUT_PATH, 'r') as z:
        filename = z.namelist()[0] 
        with z.open(filename) as f:
            raw_data = pickle.load(f)

    train_data = raw_data['train']
    all_joints = train_data['joint_2d']
    all_sources = train_data['source']

print("Grouping frames by video sequence...")


# 2. Group by Source
video_sequences = {}

for joint, source in tqdm(zip(all_joints, all_sources), total=len(all_joints)):
        if source not in video_sequences:
            video_sequences[source] = []
        video_sequences[source].append(joint)
        
print(f"✅ Found {len(video_sequences)} unique video clips.")

# 3. Slice and Normalize
processed_clips = []

print(f"Slicing into windows of {RECEPTIVE_FIELD} frames...")
for source_name, frames in tqdm(video_sequences.items()):
    frames_arr = np.array(frames)

    frames_norm = normalize_pose(frames_arr)

    clips = get_chunks(frames_norm, RECEPTIVE_FIELD, STRIDE)
        processed_clips.extend(clips)