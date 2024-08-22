from csv import reader
from Image3D import Image3D
import matplotlib.pyplot as plt


dir_csv = '/Users/varun/Documents/projects_personal/cnn/res/DL_info.csv'
dir_images = '/Users/varun/Documents/projects_personal/cnn/tmp'

dl_info = {}
with open(dir_csv) as file_obj:
    reader_obj = reader(file_obj, delimiter=",")
    heading = next(file_obj).split(",")
    index_file_name = heading.index("File_name")
    index_key_slice = heading.index("Key_slice_index")
    index_key_slice = f"{index_key_slice:03}"
    index_mask = heading.index("Bounding_boxes")
    index_windows = heading.index("DICOM_windows")
    for row in reader_obj:
        window = row[index_windows].partition(', ')
        window = (float(window[0]), float(window[2]))
        dl_info[row[index_file_name].rpartition('_')[0]] = window

window = dl_info['000010_01_01']
images = Image3D(dir_images, window[0], window[1])

xy_resolution = 512
z_resolution = 25
slices = [images.images[0].get_slice(slice/z_resolution * 180,xy_resolution) for slice in range(z_resolution)]

plt.imshow(slices, aspect='auto')
plt.colorbar()
plt.show()