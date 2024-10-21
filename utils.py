# utils.py

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def strokes_to_image(strokes, img_size=28):
    fig, ax = plt.subplots()
    for stroke in strokes:
        x = [point[0] for point in stroke]
        y = [point[1] for point in stroke]
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.plot(x, y, linewidth=5)
    ax.axis('off')
    fig.canvas.draw()
    # Convert to numpy array
    X = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    # Convert to grayscale and resize
    image = Image.fromarray(X).convert('L').resize((img_size, img_size))
    return np.array(image)
