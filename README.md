### achieved more than 96 accuracy on modelnet40 test dataset with lightweighted custome model pipeline. ###

### Projecting (added gaussian splatting) 3d pointcloud dataset into 2d image and putting them to clip vit-b/16  ###

### This image is about sampling test ### 
<img width="1200" height="600" alt="Image" src="https://github.com/user-attachments/assets/e8979f66-3afb-4717-a170-d8b0ba4eb520" />

### Other models's performance from paperswithcode ###
<img width="453" height="698" alt="image" src="https://github.com/user-attachments/assets/19f7f657-2cff-47ac-af29-1223f065617e" />

### Best training result ###
<img width="453" height="698" alt="Image" src="https://github.com/user-attachments/assets/927e9ad8-2b38-49e8-b3f1-78d7bab4fa18" />

# How to use

1) `git clone this repo`
2) `pip install -r requirements.txt`
3) `if it's first time running this model, then in inference.py, set exist_flag = True`


# additional

1) `I will make config.py to set hyperparameters easily later`
2) `torch in requirements.txt is cpu only. change it if you want to use cuda`
3) `sample test accuracy is 70% but overall performance is 95 for this pth`


citation : Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao
3D ShapeNets: A Deep Representation for Volumetric Shapes
Proceedings of 28th IEEE Conference on Computer Vision and Pattern Recognition (CVPR2015)
Oral Presentation ·  3D Deep Learning Project Webpage
