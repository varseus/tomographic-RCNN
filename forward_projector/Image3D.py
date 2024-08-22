import os
from Image2D import Image2D

class Image3D:
    def __init__(self, dir, min_window, max_window):
        self.images = []
        for filename in os.listdir(dir):
            file = os.path.join(dir, filename)
            if '.png' in file:
                self.images.append(Image2D(file, min_window, max_window))

    