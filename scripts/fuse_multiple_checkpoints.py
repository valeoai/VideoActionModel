import os
import glob
import click
from tqdm import tqdm
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

def find_deepspeed_checkpoints(root_dir, pattern="quarters*.ckpt"):
    """Find all DeepSpeed checkpoint directories matching the pattern."""
    # Search recursively in all subdirectories
    search_pattern = os.path.join(root_dir, "**/", pattern)
    checkpoint_dirs = glob.glob(search_pattern, recursive=True)
    
    # Filter to get only directories that don't have corresponding fused versions
    non_fused_checkpoints = []
    for dir_path in checkpoint_dirs:
        if not os.path.isdir(dir_path):
            continue
        fused_path = dir_path.rsplit('.ckpt', 1)[0] + '_fused.pt'
        if not os.path.exists(fused_path):
            non_fused_checkpoints.append(dir_path)
    
    return non_fused_checkpoints

def format_path(path, max_length=100):
    """Format path for display, showing ellipsis in middle if too long."""
    if len(path) <= max_length:
        return path
    head, tail = os.path.split(path)
    middle_len = max_length - len(tail) - 5  # 5 for '/.../'
    if middle_len < 10:  # if too short, just truncate from the start
        return '...' + path[-(max_length-3):]
    return f"{head[:middle_len//2]}...{head[-(middle_len//2):]}/{tail}"

def fuse_checkpoint(checkpoint_dir):
    """Fuse a single DeepSpeed checkpoint directory."""
    output_path = checkpoint_dir.rsplit('.ckpt', 1)[0] + '_fused.pt'
    
    try:
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_path)
        return True, output_path
    except Exception as e:
        print(f"\nError fusing {format_path(checkpoint_dir)}: {str(e)}")
        return False, None

@click.command()
@click.argument('root_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--pattern', '-p', default='quarters_epoch=*.ckpt',
              help='Pattern to match checkpoint directories (default: quarters_epoch=*.ckpt)')
def main(root_dir, pattern):
    """
    Fuse DeepSpeed checkpoints starting from ROOT_DIR.
    
    Searches recursively for checkpoint directories matching the pattern
    and converts them to consolidated PyTorch state dictionaries.
    """
    root_dir = os.path.abspath(root_dir)
    print(f"Searching for checkpoints in: {root_dir}")
    print(f"Using pattern: {pattern}")
    
    # Find all matching checkpoints
    checkpoints = find_deepspeed_checkpoints(root_dir, pattern)
    total_found = len(checkpoints)
    
    if not checkpoints:
        print("\nNo non-fused checkpoints found.")
        return
    
    print(f"\n{total_found} non-fused checkpoint(s) found:")
    for cp in checkpoints:
        print(f"- {format_path(cp)}")
    
    # Process each checkpoint with progress bar
    successful = 0
    failed = 0
    processed = []
    
    print("\nStarting fusion process...")
    for cp in tqdm(checkpoints, desc="Fusing checkpoints", unit="ckpt"):
        success, output_path = fuse_checkpoint(cp)
        if success:
            successful += 1
            processed.append((cp, output_path))
        else:
            failed += 1
    
    # Print summary
    print("\nFusion Complete!")
    print(f"Successfully fused: {successful}")
    print(f"Failed to fuse: {failed}")
    print(f"Total processed: {successful + failed}")
    
    if successful > 0:
        print("\nSuccessfully processed checkpoints:")
        for src, dst in processed:
            print(f"\nFrom: {format_path(src)}")
            print(f"To:   {format_path(dst)}")

if __name__ == "__main__":
    main()