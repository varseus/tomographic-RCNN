# Tomographic-RCNN

R-CNN inspired by: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/tree/master/src

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
- DeepLesion Dataset (https://nihcc.app.box.com/v/DeepLesion/folder/50715173939)
- Standard pytorch faster R-CNN model to identify lesions in 2d image slices with >90% accuracy and less than human-noticeable error in the bounding box.
  - Currently achieving net loss of 0.8142
    - TBD: Hyperparameter optimization
  - Bounding box error currently innoticeable
### 2. Deep Learning Reconstruction for Forward Projection and Back Projection
- LoDoPaB-CT dataset (https://zenodo.org/record/3384092)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9225884/
### 3. 3D Lesion Detection in Sinogram Space
- Standard pytorch faster R-CNN model to identify lesions in sinograms with >90% accuracy and less than human-noticeable error in the 3D bounding box.
