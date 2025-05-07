## Installation
```sh
pip install easydict faiss-gpu ninja open3d==0.16.0 opencv-python-headless pyyaml scikit-learn scipy tensorboard timm tqdm 
pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/unlimblue/KNN_CUDA.git#egg=knn_cuda&subdirectory=."
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## Datasets

### Anomaly-ShapeNet
Download dataset from [google drive](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet/tree/main) and extract `pcd` folder into `./data/shapenet-ad/`
```
shapenet-ad
├── ashtray0
    ├── train
        ├── ashtray0_template0.pcd
        ...
    ├── test
        ├── ashtray0_bulge0.pcd
        ...
    ├── GT
        ├── ashtray0_bulge0.txt
        ... 
├── bag0
...
...
├── vase9
```

## Training and Testing
```bash
python train_test.py configs/shapenet_dit.yaml
```

## To set window attention OFF
'''
set dit_window_size to 0 and dit_window_block_indexes to ''
'''