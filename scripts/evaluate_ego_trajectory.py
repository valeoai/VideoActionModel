import pickle

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model.evaluation import min_ade
from world_model.gpt2 import load_inference_vai0rbis
from world_model.opendv import EgoTrajectoryDataset
from world_model.utils import expand_path

save_path = "tmp/action_expert_trajectory_nuscenses_val.png"

plt.style.use("default")
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

vai0rbis = load_inference_vai0rbis(
    expand_path("$ycy_ALL_CCFRSCRATCH/test_fused_checkpoint/gpt_width768_action_dim192_fused.pt"), "cuda"
)

with open(expand_path("$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_val_data_cleaned.pkl"), "rb") as f:
    nuscenes_pickle_data = pickle.load(f)

dataset = EgoTrajectoryDataset(
    pickle_data=nuscenes_pickle_data,
    tokens_rootdir=expand_path("$ycy_ALL_CCFRSCRATCH/nuscenes_v2/tokens"),
)

loader = DataLoader(dataset, batch_size=96, shuffle=False, num_workers=10, pin_memory=True)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor("white")
# Initialize min/max for plot boundaries and yaw rates
x_min, y_min = float("inf"), float("inf")
x_max, y_max = float("-inf"), float("-inf")

num_sampling = 5

best_trajectories = []

total_loss, total_samples = 0.0, 0
iterator = tqdm(loader, "Evaluating")
for batch in iterator:
    sampled_trajectory = []
    visual_tokens = batch["visual_tokens"].to("cuda", non_blocking=True)
    commands = batch["high_level_command"].to("cuda", non_blocking=True)[:, -1:]
    for _ in range(num_sampling):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            trajectory = vai0rbis(visual_tokens, commands, torch.bfloat16)
        sampled_trajectory.append(trajectory)

    sampled_trajectory = torch.cat(sampled_trajectory, dim=1)
    ground_truth = batch["positions"].to("cuda", non_blocking=True)[:, -1]
    loss, idx = min_ade(sampled_trajectory, ground_truth, return_idx=True, reduction="sum")
    best_sampled_trajectory = sampled_trajectory[torch.arange(len(sampled_trajectory)), idx]
    total_loss += loss
    total_samples += len(ground_truth)
    iterator.set_postfix(minADE=total_loss.item() / total_samples)

    # Update plot boundaries
    x_min = min(x_min, best_sampled_trajectory[..., 0].min().item())
    x_max = max(x_max, best_sampled_trajectory[..., 0].max().item())
    y_min = min(y_min, best_sampled_trajectory[..., 1].min().item())
    y_max = max(y_max, best_sampled_trajectory[..., 1].max().item())

    for traj in best_sampled_trajectory.float().cpu():
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=1)

total_loss = total_loss.item() / total_samples
print("minADE:", total_loss)

# Add padding to the limits
padding = 0.05 * max(x_max - x_min, y_max - y_min)
ax.set_xlim(x_min - padding, x_max + padding)
ax.set_ylim(y_min - padding, y_max + padding)

# Add labels and title
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title(f"Trajectory Plot (n={len(dataset)})\nColored by Average Yaw Rate")

# Equal aspect ratio for proper visualization
ax.set_aspect("equal")

# Add grid with light gray color
ax.grid(True, linestyle="--", alpha=0.3, color="gray")

# Ensure tight layout
plt.tight_layout()

print(f"Saving plot to {save_path}...")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
