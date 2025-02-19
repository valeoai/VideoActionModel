import os
import subprocess

from vam.utils.expand_path import expand_path


def create_mp4_from_folder(folder: str, outfile: str, framerate: int = 2, quality: int = 23) -> str:
    """Create a mp4 from a folder with image files."""
    folder = expand_path(folder)
    assert os.path.exists(outdir := os.path.dirname(outfile)), f"Output folder {outdir} does not exist"
    outfile = expand_path(outfile)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-hide_banner",
        "-nostats",
        "-f",
        "image2",
        "-framerate",
        f"{framerate}",
        "-pattern_type",
        "glob",
        "-i",
        "*.{jpg,png}",
        "-c:v",
        "libx264",  # Use H.264 codec
        "-pix_fmt",
        "yuv420p",  # Standard pixel format for compatibility
        "-preset",
        "medium",  # Encoding speed preset
        "-crf",
        f"{quality}",  # Quality setting (lower = better quality, 23 is default)
        outfile,
    ]
    subprocess.run(cmd, cwd=folder)
    return outfile
