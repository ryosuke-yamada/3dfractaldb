# 3D Fractal DataBase (3D-FractalDB) 

## Summary
The repository contains a 3D Fractal Category Search, Multi-View Fractal DataBase (MV-FractalDB) and Point Cloud Fractal DataBase (PC-FractalDB) Construction in Python3.

The repository is based on the paper:<br>
Ryosuke Yamada, Ryo Takahashi, Ryota Suzuki, Akio Nakamura, Yusuke Yoshiyasu, Ryusuke Sagawa and Hirokatsu Kataoka, <br>
"MV-FractalDB: Formula-driven Supervised Learning for Multi-view Image Recognition" <br>
International Conference on Intelligent Robots and Systems (IROS) 2021 <br>
[[Project](https://ryosuke-yamada.github.io/Multi-view-Fractal-DataBase/)] 
[[PDF](https://ieeexplore.ieee.org/abstract/document/9635946)]<br>

Ryosuke Yamada, Hirokatsu Kataoka, Naoya Chiba, Yukiyasu Domae and Tetsuya Ogata<br>
"Point Cloud Pre-training with Natural 3D Structure"<br>
International Conference on Computer Vision and Pattern Recognition (CVPR) 2022 <br>
[[Project](https://ryosuke-yamada.github.io/PointCloud-FractalDataBase/)] 
[[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yamada_Point_Cloud_Pre-Training_With_Natural_3D_Structures_CVPR_2022_paper.pdf)]
[[Pre-trained Model(VoteNet)](https://github.com/ryosuke-yamada/3dfractaldb/blob/main/models/pcfractaldb_votenet_weight.tar)]<br>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-cloud-pre-training-with-natural-3d/3d-object-detection-on-sun-rgbd-val)](https://paperswithcode.com/sota/3d-object-detection-on-sun-rgbd-val?p=point-cloud-pre-training-with-natural-3d)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-cloud-pre-training-with-natural-3d/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=point-cloud-pre-training-with-natural-3d)

Ryosuke Yamada, Kensho Hara, Hirokatsu Kataoka, Koshi Makihara, Nakamasa Inoue, Rio Yokota, and Yutaka Satoh, <br>
"Formula-Supervised Visual-Geometric Pre-training" <br>
European Conference on Computer Vision (ECCV) 2024 <br>
[[Project](https://ryosuke-yamada.github.io/fdsl-fsvgp/)] 
[[PDF]()]<br>

<!-- Run the python script ```render.sh```, you can get 3D fractal models and multi-view fractal images. -->

<!-- ## Prerequisites
- Anaconda
- Python 3.9+ -->

<!-- ## Installation
1. Create conda virtual environment.
```
$ conda create -n mvfdb python=3.9 -y
$ conda activate mvfdb
```
2. Install requirement modules
```
$ conda install -c conda-forge openexr-python
$ pip install -r requirements.txt
``` -->

## MV-FractalDB Construction ([README](https://github.com/ryosuke-yamada/3dfractaldb/blob/main/mvfractaldb/README.md))
```
$ cd mvfractaldb/
$ bash MVFractalDB_render.sh
```

## PC-FractalDB Construction ([README](https://github.com/ryosuke-yamada/3dfractaldb/blob/main/pcfractaldb/README.md))
```
$ cd pcfractaldb
$ bash PCFractalDB_render.sh
```

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{yamada2021mv,
  title={MV-FractalDB: Formula-driven Supervised Learning for Multi-view Image Recognition},
  author={Yamada, Ryosuke and Takahashi, Ryo and Suzuki, Ryota and Nakamura, Akio and Yoshiyasu, Yusuke and Sagawa, Ryusuke and Kataoka, Hirokatsu},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2076--2083},
  organization={IEEE}
}
```
```bibtex
@InProceedings{Yamada_2022_CVPR,
    author    = {Yamada, Ryosuke and Kataoka, Hirokatsu and Chiba, Naoya and Domae, Yukiyasu and Ogata, Tetsuya},
    title     = {Point Cloud Pre-Training With Natural 3D Structures},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21283-21293}
}
```

## Terms of use
The authors affiliated in National Institute of Advanced Industrial Science and Technology (AIST), Tokyo Denki University (TDU), Waseda University and Keio University are not responsible for the reproduction, duplication, copy, sale, trade, resell or exploitation for any commercial purposes, of any portion of the datas and any portion of derived the data. In no event will we be also liable for any other damages resulting from this data or any derived data.
