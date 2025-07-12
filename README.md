achieved more than 96 accuracy on modelnet40 test dataset with lightweighted custome model pipeline.
Projecting (added gaussian splatting) 3d pointcloud dataset into 2d image and putting them to clip vit-b/16  

### This image is about sampling test ### 
<img width="1200" height="600" alt="Image" src="https://github.com/user-attachments/assets/e8979f66-3afb-4717-a170-d8b0ba4eb520" />

### Other models's performance from paperswithcode ###
<img width="1785" height="567" alt="Image" src="https://github.com/user-attachments/assets/f52f4cba-f5fa-493a-a978-565d264e1d34" />

### Best training result ###
<img width="453" height="698" alt="Image" src="https://github.com/user-attachments/assets/927e9ad8-2b38-49e8-b3f1-78d7bab4fa18" />

# How to use

1) `python==3.10` 
2) `git clone this repo`
3) `cd 3d-vlm-gaussian-splatting-classification-on-modelnet40` 
4) `pip install -r requirements.txt`
5) `python -m inference.py`
6) `if it's first time running this model, then in config.py, set exist_flag = True`
7) `if you want to do inference on more dataset, you can set max_samples in config.py`


# additional

1) `torch in requirements.txt is cpu only. change it if you want to use cuda`
2) if you want to use cuda version, delete torch in requirements.txt and do (3)
3) `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<version you want e.g) 118 if cuda 11.8>`
4) `sample test accuracy is 70% and overall validation performance for test datasrt is 95 for this pth`
5) you can run this even in only cpu env


# citation 
Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao
3D ShapeNets: A Deep Representation for Volumetric Shapes
Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)
Oral Presentation ·  3D Deep Learning Project Webpage
