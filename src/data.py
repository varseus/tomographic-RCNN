import torch
from PIL import Image
from csv import reader
import numpy as np
from os.path import isfile


# Dataset for the key image in each scan from DeepLesion Dataset
class CTDataset(torch.utils.data.Dataset):
    def __init__(self, transform, dir_csv="/Volumes/varunT7/DL_info.csv", dir_images="/Volumes/varunT7/Images_Train"):
        print("Reading data from " + dir_csv)
        images = {} # dictionary of scans to output | {scan_index : [dir_image, mask], ...} ### consider changing this to a list for less time efficiency but more space efficiency
        targets = {}
        with open(dir_csv) as file_obj:
            reader_obj = reader(file_obj, delimiter=',')
            heading = next(file_obj).split(",")
            index_file_name = heading.index("File_name")
            index_key_slice = heading.index("Key_slice_index")
            index_key_slice = f"{index_key_slice:03}"
            index_mask = heading.index("Bounding_boxes")
            i = 0
            for row in reader_obj:
                dir_image = dir_images + "/" + "/".join(row[index_file_name].rsplit("_", 1))
                if isfile(dir_image):
                    target = {}
                    target["boxes"] = torch.as_tensor([[float(i) for i in row[index_mask].split(", ")]], dtype=torch.float16)
                    target["iscrowd"] = torch.as_tensor([0], dtype=torch.bool)
                    target["labels"] = torch.as_tensor([1], dtype=torch.int64)
                    target["image_id"] = torch.as_tensor([i], dtype=torch.int64)
                    images[str(i)] = dir_image
                    targets[str(i)] = target
                    i+=1

        self.images = images
        self.targets = targets
        self.transform = transform

        print("Found " + str(len(images)) + " training images")
                # with Image.open() as image:
                #     image = Image.fromarray((np.array(image) - 32768) * 256 / 2000 + (256/2))


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, i):
        i = str(i)
        with Image.open(self.images[i]) as image:
            image = Image.fromarray((np.array(image) - 32768) * 256 / 2000 + (256/2))

            if self.transform is not None:
                image, self.targets[i] = self.transform(image, self.targets[i])

            return image, self.targets[i]
