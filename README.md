# Tomographic-RCNN
Code inspired by: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/tree/master/src
---
## Exigence
This project aims to identify 3D bounding boxes for lesions in CT sinogram data, where the buonding box coordinates
correlate to the 8 bounding pixel coordinates of the lesion in the reconstructed image. Success will be defined by >90%
accuracy in identifying lesions with less than human-noticeable error in the bounding box.

Pre-reconstruction analysis runs counter to the current research emphasis on sinogram/image synthesis.
Human-analysis of CT data for patient diagnosis requires large data corrections and high quality back projections,
such as to minimize human error. However, neural networks can directly process raw CT data, which may reduce
error and resource expenditure. Analysis of sinogram data effectively automates patient diognosis further upstream, 
which is a yet unexplored technique.

---
## Outline
### 1. 2D Lesion Detection in CT Image Slices
### 2. Deep Learning Reconstruction for Forward Projection and Back Projection
### 3. 3d Lesion Detection in Sinogram Space
