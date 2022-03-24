# Multi-View Fractal DataBase (MV-FractalDB) 

Run the python script ```render.sh```, you can get 3D fractal models and multi-view fractal images.

<!-- ## Prerequisites
- Anaconda
- Python 3.9+ -->

## Installation
### Prepare environment
1. Create conda virtual environment.
```
$ conda create -n mvfdb python=3.9 -y
$ conda activate mvfdb
```

2. Install requirement modules
```
$ conda install -c conda-forge openexr-python
$ pip install -r requirements.txt
```

## Construct MV-FractalDB
1. Search fractal category and create a 3D fractal model
```
$ cd 3dfractal_render
$ bash render.sh
```

2. Render multi-view image
```
$ cd image_render
$ python render.py
```