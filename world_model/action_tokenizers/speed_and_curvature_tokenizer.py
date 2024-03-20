import torch
import torch.nn as nn
from einops import reduce, rearrange
import numpy as np
import torch
import ast



class SpeedCurvatureTokenizer(nn.Module):
    def __init__(self, vocab_size: int = 5, centroids = None, data_min=None, data_max=None, centroids_file=None):
        """
        A class to tokenize speed-curvature action using KMeans clustering.

        Args:
            vocab_size: Number of clusters for K-means.
            centroids: Initial centroids for K-means clustering.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        
        self.register_buffer('data_min', torch.tensor(data_min))
        self.register_buffer('data_max', torch.tensor(data_max))
        
        if centroids is not None:
            self.register_buffer('centroids', torch.tensor(centroids))
        
        if centroids_file is not None:
            with open(centroids_file, 'r') as file:
                data = file.read()
            centroids = ast.literal_eval(data)
            self.register_buffer('centroids', torch.tensor(centroids))

    def min_max_normalization(self, data):
        return (data - self.data_min) / (self.data_max - self.data_min)
    
    def forward(self, ego_to_world_rot, ego_to_world_tran, timestamps, **kwargs):
        b, t, *_ =  ego_to_world_rot.shape
        
        yaws = quaternion_to_yaw(ego_to_world_rot)
        speeds, curvatures = compute_signed_speed_and_curvature(ego_to_world_tran, yaws)
        
        data = torch.stack([speeds, curvatures], dim=-1)

        normalized_data = self.min_max_normalization(data)

        normalized_data = rearrange(normalized_data, 'b t c -> b t 1 c')
        squared_diff = torch.pow(normalized_data - self.centroids, 2)
        distances = reduce(squared_diff, 'b t k c -> b t k', 'sum').sqrt()
        action_tokens = distances.argmin(-1, keepdim=True)   

        return action_tokens
    
    
    def get_params(self):
        return {
            'vocab_size': self.vocab_size,
            'centroids': self.centroids.tolist(),
            'data_min': self.data_min.tolist(),
            'data_max': self.data_max.tolist()
        }

    def augment_data(self, speeds: np.array, curvatures: np.array) -> tuple[np.array, np.array]:
        curvatures_abs = np.abs(curvatures)
        curvatures_neg = -curvatures_abs
        aug_speeds = np.hstack([speeds, speeds])
        aug_curvatures = np.hstack([curvatures_abs, curvatures_neg])
        return aug_speeds, aug_curvatures
    
    def set_min_max_from_data(self, data):
        self.data_min = data.min(axis=0)
        self.data_max = data.max(axis=0)
        
    def learn(self, speeds: np.array, curvatures: np.array):
        """
        Learns the cluster centers from the provided speeds and curvatures data.

        Args:
            speeds: Speed values.
            curvatures: Curvature values.
        """
        from sklearn.cluster import KMeans
        
        aug_speeds, aug_curvatures = self.augment_data(speeds, curvatures)
        aug_curvatures_scaled = aug_curvatures

        data = np.column_stack([aug_speeds, aug_curvatures_scaled])

        self.set_min_max_from_data(data)
        normalized_data = self.min_max_normalization(data)
        
        kmeans = KMeans(n_clusters=self.vocab_size, random_state=42)
        kmeans.fit(normalized_data)
        
        self.centroids = kmeans.cluster_centers_



def get_transformation_matrix(rotation, translation):
    """
    Create a 4x4 transformation matrix from a batch of 3x3 rotation matrices and translation vectors.
    
    Parameters:
        rot: A tensor of shape [B, T, 3, 3] containing the rotation matrices.
        trans: A tensor of shape [B, T, 3] containing the translation vectors.
    
    Returns:
        A tensor of shape [B, T, 4, 4] containing the 4x4 transformation matrices.
    """
    B, S, *_ = rotation.shape
    
    # concat the translation column at the most right position
    translation = rearrange(translation, '... -> ... 1')
    upper_matrix = torch.cat([rotation, translation], dim=-1)
    
    # concat the [0, 0, 0, 1] line at the bottom
    ones_vector = torch.zeros((B, S, 1, 4), device=rotation.device, dtype=rotation.dtype)
    ones_vector[..., -1] = 1.0
    transformation_matrix = torch.cat([upper_matrix, ones_vector], dim=-2)
    
    return transformation_matrix



def quaternion_to_yaw(quaternions: torch.Tensor):
    """    
    This function calculates the yaw angle, defined here as the angle between the vehicle's forward direction 
    (y-axis in nuscenes) and the global y-axis, measured in the horizontal plane and around the upward-pointing z-axis. 
    A positive yaw indicates a counter-clockwise rotation when viewed from above, aligning with the right-hand rule.
    
    Parameters:
        quaternions: A tensor of shape [B, T, 4] representing a batch of quaternions for each pose in the sequence. 
        Quaternions encode the rotation of the vehicle's reference w.r.t. the global frame.
        Each quaternion is given as [w, x, y, z], where w is the scalar part, 
        and x, y, z represent the vector part of the quaternion.
    
    Returns:
        torch.Tensor: A tensor of shape [B, T] with the yaw angles for each quaternion. The yaw is calculated in radians 
        and ranges from -π to π, describing the rotation around the z-axis.
    
    Notes:
        - It is assumed that the quaternions are normalized.
        - A positive yaw corresponds to a counter-clockwise rotation when looking down from above the vehicle.
        - This function adheres to the aerospace convention for yaw, pitch, and roll, where rotations are applied 
        sequentially around the z, y, and x-axes, respectively.
    """
    # Extract components of quaternion
    w, x, y, z = quaternions.unbind(-1)
    
    # Compute yaw (rotation around z-axis) from quaternion
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return yaw



def compute_signed_speed_and_curvature(ego_to_world_tran: torch.Tensor, yaws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes curvature and signed speed (incorporating direction) for a batched sequence of vehicle poses.

    Arguments:
    - ego_to_world_tran: A tensor of shape [B, T, 3] representing batched sequences of translations (x, y, z) for each vehicle.
    - yaws: A tensor of shape [B, T] representing batched sequences of yaw angles, indicating the vehicle's orientation.

    Returns:
    - Signed Speeds: A tensor of shape [B, T-1], representing the speeds with sign indicating the direction of movement (positive for forward, negative for backward, zero for no movement).
    - Curvatures: A tensor of shape [B, T-1], representing the vehicle's turning behavior based on the change in yaw over the distance traveled.
    
    Assumptions:
    - The time step between each pose is constant (0.5 seconds).
    - The signed speed is calculated as the Euclidean distance between positions at consecutive timesteps divided by the time interval, with the sign indicating the direction (forward or backward).
    - Curvature is defined as the change in yaw angle per unit of distance traveled.
    - Direction of movement is determined by the dot product of the vehicle's orientation vector with the movement vector between consecutive positions.

    Note:
    - A curvature of 0 implies a straight trajectory, while non-zero values indicate turning (positive for right, negative for left).
    - Signed speed provides insight into the speed and direction of the vehicle's movement.
    """
    # Calculate the difference in position to determine speed
    delta_tran = ego_to_world_tran[:, 1:, :] - ego_to_world_tran[:, :-1, :]
    speeds = torch.norm(delta_tran, dim=-1) / 0.5  # Speed calculation assuming Δt = 0.5 seconds
    
    # Calculate the change in yaw angles to determine curvature
    delta_yaw = yaws[:, 1:] - yaws[:, :-1]
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize delta_yaw to the range [-pi, pi]
    distances = torch.norm(delta_tran, dim=-1)
    curvatures = delta_yaw / (distances + 1e-10)  # Calculate curvature, adding epsilon to avoid division by zero

    curvatures[distances == 0.] = 0.
    # = 15cm/0.5s = 30cm/s = 1.08km/h
    curvatures[speeds < 0.15] = 0.
    
    # Calculate orientation vectors based on yaws for direction determination
    orientation_vectors = torch.stack((torch.cos(yaws[:, :-1]), torch.sin(yaws[:, :-1]), torch.zeros_like(yaws[:, :-1])), dim=-1)
    
    # Dot product between orientation vectors and movement vectors to determine direction
    dot_products = (orientation_vectors * delta_tran).sum(dim=-1)
    directions = torch.sign(dot_products)  # Sign of dot product indicates direction (1 for forward, -1 for backward)
    directions[dot_products == 0] = 0  # 0 indicates no movement

    # Calculate signed speeds by multiplying speeds with their respective directions
    signed_speeds = speeds * directions  # Signed speeds indicate magnitude and direction of movement

    return signed_speeds, curvatures
