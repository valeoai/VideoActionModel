import os
import subprocess
import tempfile
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from vam.utils.expand_path import expand_path


def _create_text_overlay(text: str, width: int = 110, height: int = 30, fontsize: int = 20) -> str:
    """Create a PNG overlay with text."""
    # Create temporary file
    fd, overlay_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    # Create a white background image with text
    img = Image.new("RGBA", (width, height), (255, 255, 255, 180))  # White with transparency
    draw = ImageDraw.Draw(img)

    # Try to use a system font
    try:
        font = ImageFont.truetype("Arial", fontsize)
    except IOError:
        try:
            # Try another common font
            font = ImageFont.truetype("DejaVuSans", fontsize)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()

    # Draw black text
    draw.text((width // 2, height // 2), text, fill=(0, 0, 0), font=font, anchor="mm")
    img.save(overlay_path)

    return overlay_path


def create_mp4_from_folder(
    folder: str, outfile: str, framerate: int = 2, quality: int = 23, overlay: Optional[str] = None, **kwargs
) -> str:
    """Create a mp4 from a folder with image files."""
    folder = expand_path(folder)
    outfile = expand_path(outfile)
    assert os.path.exists(outdir := os.path.dirname(outfile)), f"Output folder {outdir} does not exist"

    # Create overlay image with text
    overlay_image = _create_text_overlay(overlay, **kwargs) if overlay else None

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
        "-i" if overlay else "",
        overlay_image if overlay else "",
        "-filter_complex" if overlay else "",
        "overlay=10:10" if overlay else "",
        "-c:v",
        "libx264",  # Use H.264 codec
        "-pix_fmt",
        "yuv420p",  # Standard pixel format for compatibility
        "-preset",
        "medium",  # Encoding speed preset
        "-crf",
        f"{quality}",  # Quality setting (lower = better quality, 23 is default)
        "-y",
        outfile,
    ]

    cmd = [c for c in cmd if c != ""]  # Remove empty strings
    subprocess.run(cmd, cwd=folder)
    return outfile


def concatenate_mp4(video_files: List[str], output_file: str) -> str:
    """Concatenate multiple video files."""
    # Create temporary file list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for video in video_files:
            f.write(f"file '{expand_path(video)}'\n")
        filelist = f.name

    output_file = expand_path(output_file)
    assert os.path.exists(outdir := os.path.dirname(output_file)), f"Output folder {outdir} does not exist"

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-hide_banner",
        "-nostats",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        filelist,
        "-c",
        "copy",
        "-y",
        output_file,
    ]

    subprocess.run(cmd)

    # Clean up
    os.remove(filelist)

    return output_file


if __name__ == "__main__":
    create_mp4_from_folder(
        "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/context",
        "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/real_overlay.mp4",
        overlay="Real",
    )

    create_mp4_from_folder(
        "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/generated",
        "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/generated_overlay.mp4",
        overlay="Generated",
        width=110,
    )

    concatenate_mp4(
        [
            "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/real_overlay.mp4",
            "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/generated_overlay.mp4",
        ],
        "$cya_ALL_CCFRSCRATCH/qual_results/vavim_l/nuscenes/28/concatenated.mp4",
    )
