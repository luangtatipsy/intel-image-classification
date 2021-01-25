import os
import textwrap

import matplotlib.pyplot as plt
from PIL import Image


def display_images(
    paths,
    category,
    columns=10,
    width=30,
    height=5,
    max_images=20,
    label_wrap_length=50,
    label_font_size=8,
    save_figure=True,
):
    images = [Image.open(path) for path in paths]

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images = images[0:max_images]

    height = max(height, int(len(images) / columns) * height)
    figure = plt.figure(figsize=(width, height))
    figure.suptitle(category.title(), fontsize=28, fontweight="bold")

    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)

        if hasattr(image, "filename"):
            title = image.filename
            if title.endswith("/"):
                title = title[0:-1]

            title = os.path.basename(title)
            title = textwrap.wrap(title, label_wrap_length)
            title = "\n".join(title)
            plt.title(title, fontsize=label_font_size)

    if save_figure:
        plt.savefig(os.path.join("results", "figures", f"{category}.png"))
