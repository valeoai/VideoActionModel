#!/bin/bash
#SBATCH --job-name=remove_dir
#SBATCH -A ycy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-gc
#SBATCH --time=05:00:00
#SBATCH --output=remove_dir-%j.out
#SBATCH --error=remove_dir-%j.out

# Set the target directory
TARGET_DIR=$fzh_ALL_CCFRSCRATCH/OpenDV_processed/frames512

# Function to remove a directory
remove_dir() {
    local dir="$1"
    echo "Removing directory: $dir"
    rm -rf "$dir"
}

# Export the function so it can be used by xargs
export -f remove_dir

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR does not exist."
    exit 1
fi

# Change to the target directory
cd "$TARGET_DIR" || exit 1

# Find all directories and remove them in parallel
find . -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -P $SLURM_CPUS_PER_TASK -I {} bash -c 'remove_dir "{}"'

echo "All directories in $TARGET_DIR have been processed."
