from math import floor, dist, radians, sin, cos
import numpy as np
from PIL import Image

class Image2D:
    def __init__(self, file, min_window, max_window):
        with Image.open(file) as image:
            self.image = np.array(image, dtype=np.int32)
            self.image = ((self.image - 32768 - min_window) / (max_window - min_window))
            self.image = np.min(
                (
                    np.ones(self.image.shape),
                    np.max(
                        (
                            np.zeros(self.image.shape),
                            self.image,
                        ),
                        axis=0,
                    ),
                ),
                axis=0,
            )

    def get_pixel_interpolate(self, x, y):
        top_left = floor(x)
        top_right = floor(y)

        points = [(top_left, top_right),
                  (top_left+1, top_right),
                  (top_left, top_right + 1),
                  (top_left + 1, top_right + 1)]

        if (x%1 == 0):
            points = [(int(x), point[1]) for point in points]
    
        if (y%1 == 0):
            points = [(point[0], int(y)) for point in points]
    

        distances_to_points = [dist((x,y), point) for point in points]
        normalization_factor = np.sum(distances_to_points)

        if (normalization_factor <= 0):
            return self.image[int(x)][int(y)]

        distances_to_points = [1-distance/normalization_factor for distance in distances_to_points]
        normalization_factor = np.sum(distances_to_points)
        distances_to_points = [distance/normalization_factor for distance in distances_to_points]
        return np.sum([self.image[point[0]][point[1]]*distance for point,distance in zip(points, distances_to_points)])
    
    def get_slice(self, theta, resolution):
        # Assumes image is square
        print(f'Forward Projecting at {theta} degrees')
        theta = radians(theta)
        x_vector = cos(theta)
        y_vector = sin(theta)
        x_n_vector = y_vector
        y_n_vector = - x_vector

        pointspace = np.linspace(0, len(self.image)-2, resolution)
        center = pointspace[len(pointspace)//2]
        pointspace -= center

        points = [zip(np.linspace(
            point*x_vector+center - np.sqrt(pointspace[0]**2 - point**2)*x_n_vector, 
            point*x_vector+center + np.sqrt(pointspace[0]**2 - point**2)*x_n_vector, 
            resolution),np.linspace(
            point*y_vector+center - np.sqrt(pointspace[0]**2 - point**2)*y_n_vector, 
            point*y_vector+center + np.sqrt(pointspace[0]**2 - point**2)*y_n_vector, 
            resolution)) for point in pointspace]
        
        pixels = [[self.get_pixel_interpolate(x,y) for x,y in line] for line in points]

        pixels = np.sum(pixels, axis=1) / resolution * 256

        return pixels
    