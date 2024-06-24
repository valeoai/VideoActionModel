h = """
This script produces a pickle containing the data from Nuscenes in one python dict with the following form: 

{
    'scene': {
        name: <string> --- identifier of the scene the sample is part of 'scene-0061',
        description: <string> --- e.g., 'rain'
    }
    'CAM_FRONT_LEFT': {
        'file_path': path to image file from dataroot,
        'timestamp': <int> -- Unix time stamp at which the data (image and ego pose) have been recorded for this sensor,
        'intrinsic': <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras,
        'sensor_to_ego_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z,
        'sensor_to_ego_tran': <float> [3] -- Coordinate system origin in meters: x, y, z,
        'ego_to_world_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z, 
        'ego_to_world_tran': <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0,
    },
    'CAM_FRONT': {...},
    'CAM_FRONT_RIGHT': {...},
    'CAM_BACK_LEFT': {...},
    'CAM_BACK': {...},
    'CAM_BACK_RIGHT': {...},
    'LIDAR_TOP': {
        'file_path': path to pointcloud file from dataroot,
        'timestamp': <int> -- Unix time stamp at which the data (pointcloud and ego pose) have been recorded for this sensor,
        'sensor_to_ego_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z,
        'sensor_to_ego_tran': <float> [3] -- Coordinate system origin in meters: x, y, z,
        'ego_to_world_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z, 
        'ego_to_world_tran': <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0,
    },
    'annotations': [
        {
            'category': 'vehicle.car',
            'instance_token': <string> -- unique identifier of theo object (consistent across frames),
            'box_size': <float> [3] -- Bounding box size in meters as width, length, height,
            'box_center':  <float> [3] -- Bounding box location in meters as center_x, center_y, center_z,
            'box_orientation': <float> [4] -- Bounding box orientation as quaternion: w, x, y, z,
            'visibility': <int> -- ranging from to 1-4,
            'num_lidar_pts': <int>,
            'num_radar_pts': <int>,
            'attributes': [
                {
                    'name': <string> -- e.g., 'vehicle.moving',
                    'description': <string> -- e.g, 'Vehicle is moving.'
                },
                ...
            ]
        },
        ...
    ]
}

Example usage:
------------------------------------------------------------

python ./gen_nuscenes_pickle.py \
/data_root/nuscenes  \
trainval \
/path_to_output_folder/nuscenes_pickle/trainval_data.pkl

python ./gen_nuscenes_pickle.py \
/data_root/nuscenes  \
mini \
/path_to_output_folder/nuscenes_pickle/mini_data.pkl
    
"""

import click
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pathlib import Path
import pickle
from tqdm import tqdm

def get_scenes(nuscenes_version, split):
    """
    Args:
        nuscenes_version: 'trainval' or 'mini'
        split: 'train' or 'val'
    """
    
    # filter by scene split
    split = ('mini_' if nuscenes_version=='mini' else '') + split
    scenes = create_splits_scenes()[split]

    return scenes

def prepro(nusc, scenes):
    samples = [samp for samp in nusc.sample]

    # remove samples that aren't in given split
    samples = [samp for samp in samples if
               nusc.get('scene', samp['scene_token'])['name'] in scenes]

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    return samples

def get_data(nusc, sample, cams):
    """
    Extract the sensors (file paths, intrinsics, pose, etc.) information from a given sample
    """
    
    scene = nusc.get('scene', sample['scene_token'])
    
    return {
        'scene': {
            'name': scene['name'],   
            'description': scene['description']
        }, 
        **get_cams_data(nusc, sample, cams),
        'LIDAR_TOP': get_lidar_data(nusc, sample),
        'annotations': get_annotations(nusc, sample)
    }

def get_cams_data(nusc, sample, cams):
        
    cams_data = {}

    for cam_name in cams:            
        cam_data = nusc.get('sample_data', sample['data'][cam_name])
        cam_calibration = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # note there is also cam_ego_pose['timestamp'] but it is equal to cam_data['timestamp']

        cams_data[cam_name] = {
            'file_path': cam_data['filename'],
            'timestamp': cam_data['timestamp'],
            'intrinsic': cam_calibration['camera_intrinsic'], # <float> [3, 3]
            'sensor_to_ego_rot': cam_calibration['rotation'], # <float> [4] Coordinate system orientation as quaternion: w, x, y, z.
            'sensor_to_ego_tran': cam_calibration['translation'], # <float> [3] Coordinate system origin in meters: x, y, z.
            'ego_to_world_rot': cam_ego_pose['rotation'], 
            'ego_to_world_tran': cam_ego_pose['translation'],
        }
    return cams_data

def get_lidar_data(nusc, sample):

    # Retrieve transformation matrices for reference point cloud.
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_calibration = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    return {
            'file_path': lidar_data['filename'],
            'timestamp': lidar_data['timestamp'],
            'sensor_to_ego_rot': lidar_calibration['rotation'],
            'sensor_to_ego_tran': lidar_calibration['translation'],
            'ego_to_world_rot': lidar_ego_pose['rotation'],
            'ego_to_world_tran': lidar_ego_pose['translation'],
    }

def get_annotations(nusc, sample):
    annotations = []
    for sample_annotation_token in sample['anns']:
        annotation_data = nusc.get('sample_annotation', sample_annotation_token)
        annotations.append({
            'category': annotation_data['category_name'],
            'instance_token': annotation_data['instance_token'],
            'box_size': annotation_data['size'], # <float> [3] -- Bounding box size in meters as width, length, height.
            'box_center': annotation_data['translation'], # <float> [3] -- Bounding box location in meters as center_x, center_y, center_z.
            'box_orientation': annotation_data['rotation'], # <float> [4] -- Bounding box orientation as quaternion: w, x, y, z.
            'visibility': int(annotation_data['visibility_token']),
            'num_lidar_pts': int(annotation_data['num_lidar_pts']),
            'num_radar_pts': int(annotation_data['num_radar_pts']),
            'attributes': get_annotation_attributes(nusc, annotation_data) # e.g. "vehicle.parked"
        })        
    return annotations

def get_annotation_attributes(nusc, annotation_data):
        """
        A 3D box annotation may be given one or multiple `attribute` tokens, for example:

            {
                'token': '58aa28b1c2a54dc88e169808c07331e3',
                'name': 'vehicle.parked',
                'description': 'Vehicle is stationary (usually for longer duration) with no immediate intent to move.'
            }

            this function returns a list of dict containing the `name` and `description` fields 
            for each attribute of a given annotation
        
        Args:
            annotation_data: the output of nusc.get('sample_annotation', annotation_token)
        
        Returns:
            list of dict
        """
        
        attributes = []
        for attribute_token in annotation_data['attribute_tokens']:
            attribute_data = nusc.get("attribute", attribute_token).copy() # copy to not modify the original nusc object
            attribute_data.pop('token') # discard `token` entry, only `name` and `description` are needed
            attributes.append(attribute_data)
        
        return attributes


@click.command()
@click.argument('nuscenes_root_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('nuscenes_version', type=str)
@click.argument('output_path', type=click.Path(file_okay=True))
def main(nuscenes_root_dir, nuscenes_version, output_path):
    """
    This script produces a pickle containing the data from Nuscenes in one python dict.
    
    \b
    NUSCENES_ROOT_DIR is the root folder of Nuscenes in which you find the folders `v1.0-trainval`, `samples`, `sweeps`  
    NUSCENES_VERSION  is either 'trainval' or 'mini'
    OUTPUT_PATH       is the full path to the pickle that will be produced, e.g., full_path/trainval_data.pkl
    """
    
    # Prints for user
    print("NUSCENES DIR:     ", nuscenes_root_dir)
    print("NUSCENES VERSION: ", nuscenes_version)
    print("OUTPUT PATH:      ", output_path)
    
    # Input sanity checks        
    if nuscenes_version not in ['trainval', 'mini']:
        raise ValueError(f"The `split` argument must either be 'trainval' or 'mini'. Current value: {nuscenes_version}")
    
    
    print('Loading Nuscenes data....')
    nusc = NuScenes(version='v1.0-{}'.format(nuscenes_version),
                    dataroot=nuscenes_root_dir,
                    verbose=True)
    
    print('Preprocessing data....')
    infos = {}
    for split in ['train', 'val']:
        print(f'Processing split <{split}>...')
        scenes = get_scenes(nuscenes_version, split)
        samples = prepro(nusc, scenes)

        CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']

        split_infos = []
        for sample in tqdm(samples):
            split_infos.append(get_data(nusc, sample, CAMS))
            
        infos[split] = split_infos
         
    print('Saving data infos in pickle....')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # create parent directories if they don't exist
    with open(output_path, 'wb') as f:
        pickle.dump(infos, f)
    print(f"Pickle saved: {output_path}")

if __name__ == '__main__':
    main()