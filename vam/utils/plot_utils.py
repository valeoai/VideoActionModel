import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_images(images: np.ndarray, num_rows: int, num_cols: int) -> plt.Figure:
    fig = plt.figure(figsize=(20, 5))

    idx = 0
    for _ in range(num_rows):
        for _ in range(num_cols):
            if idx >= len(images):
                break
            plt.subplot(num_rows, num_cols, idx + 1)
            plt.imshow(images[idx])
            plt.axis("off")
            idx += 1

    plt.show()
    return fig
