import pickle
import os
from tqdm import tqdm
import random

data_root_dir = '/lustre/fsn1/projects/rech/ycy/commun/OpenDV_tokenized/frames512/VQ_ds16_16384_llamagen'
train_video_list_path = '/lustre/fsn1/projects/rech/ycy/commun/OpenDV_sharded/train_videos.txt'
val_video_list_path = '/lustre/fsn1/projects/rech/ycy/commun/OpenDV_sharded/val_videos.txt'

def check_video_existence(data_root_dir, video_list):
    existing_videos = []
    missing_videos = []
    for video in tqdm(video_list, desc="Checking video existence"):
        video_path = os.path.join(data_root_dir, video)
        if os.path.exists(video_path):
            existing_videos.append(video)
        else:
            missing_videos.append(video)
    return existing_videos, missing_videos

def get_windows(data_root_dir, video_list):
    video_frames = {}
    for video_id in tqdm(video_list, desc="Processing videos"):
        video_dir = os.path.join(data_root_dir, video_id)
        frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.npy')])
        video_frames[video_id] = frames
    sequence_length = 20
    video_windows = []
    for video_id, frames in tqdm(video_frames.items(), desc="Creating windows"):
        for start_idx in range(len(frames) - sequence_length + 1):
            video_windows.append((video_id, start_idx))
    return video_windows

def pickle_train_windows(data_root_dir, video_list_path, name):
    with open(video_list_path, 'r') as f:
        video_list = [line.strip() for line in f.readlines()]
    print(f"Processing {name} videos...")
    video_list, _ = check_video_existence(data_root_dir, video_list)
    video_windows = get_windows(data_root_dir, video_list)
    
    print(f"Shuffling {name} windows...")
    random.shuffle(video_windows)
    
    total_windows = len(video_windows)
    split1 = int(total_windows * 0.33)
    split2 = int(total_windows * 0.67)
    
    print(f"Splitting and pickling {name} windows...")
    with open(f'{name}_part1.pkl', 'wb') as file:
        pickle.dump(video_windows[:split1], file)
    with open(f'{name}_part2.pkl', 'wb') as file:
        pickle.dump(video_windows[split1:split2], file)
    with open(f'{name}_part3.pkl', 'wb') as file:
        pickle.dump(video_windows[split2:], file)
    
    print(f"Finished processing {name} videos")
    del video_list
    del video_windows

def pickle_val_windows(data_root_dir, video_list_path, name):
    with open(video_list_path, 'r') as f:
        video_list = [line.strip() for line in f.readlines()]
    print(f"Processing {name} videos...")
    video_list, _ = check_video_existence(data_root_dir, video_list)
    video_windows = get_windows(data_root_dir, video_list)
    
    print(f"Pickling {name} windows...")
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(video_windows, file)
    
    print(f"Finished processing {name} videos")
    del video_list
    del video_windows

print("Processing and splitting train videos...")
pickle_train_windows(data_root_dir, train_video_list_path, 'train_opendv_windows')
print("Processing validation videos...")
pickle_val_windows(data_root_dir, val_video_list_path, 'val_opendv_windows')
print("All processing complete.")