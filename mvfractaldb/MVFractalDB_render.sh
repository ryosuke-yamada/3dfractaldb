#! /bin/bash
variance_threshold=0.05
<<<<<<< HEAD
numof_category=1
param_path='../dataset/MVFractalDB-'${numof_category}'/3DIFS_params'
model_save_path='../dataset/MVFractalDB-'${numof_category}'/3Dmodels'
image_save_path='../dataset/MVFractalDB-'${numof_category}'/images'
=======
numof_category=1000
param_path='./../MVFractalDB/3DIFS_params/MVFractalDB-'${numof_category}
model_save_path='./../MVFractalDB/3Dmodels/MVFractalDB-'${numof_category}
image_save_path='./../MVFractalDB/images/MVFractalDB-'${numof_category}
>>>>>>> 5f4ffd177d241a71684006b2ca9ad19357a675df

# Parameter search
python 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python 3dfractal_render/instance.py --load_root ${param_path} --save_root ${model_save_path} --classes ${numof_category}

# Render Multi-view images
<<<<<<< HEAD
python image_render/render.py --load_root ${model_save_path} --save_root ${image_save_path}
variance_threshold=0.05
numof_category=1
param_path='../dataset/MVFractalDB-'${numof_category}'/3DIFS_params'
model_save_path='../dataset/MVFractalDB-'${numof_category}'/3Dmodels'
image_save_path='../dataset/MVFractalDB-'${numof_category}'/images'

# Parameter search
python 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python 3dfractal_render/instance.py --load_root ${param_path} --save_root ${model_save_path} --classes ${numof_category}

# Render Multi-view images
python image_render/render.py --load_root ${model_save_path} --save_root ${image_save_path}
variance_threshold=0.05
numof_category=1
param_path='../dataset/MVFractalDB-'${numof_category}'/3DIFS_params'
model_save_path='../dataset/MVFractalDB-'${numof_category}'/3Dmodels'
image_save_path='../dataset/MVFractalDB-'${numof_category}'/images'

# Parameter search
python 3dfractal_render/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category} --save_root=${param_path}

# Generate 3D fractal model
python 3dfractal_render/instance.py --load_root ${param_path} --save_root ${model_save_path} --classes ${numof_category}

# Render Multi-view images
python image_render/render.py --load_root ${model_save_path} --save_root ${image_save_path}
=======
python image_render/render.py --load_root ${model_save_path} --save_path ${image_save_path}
>>>>>>> 5f4ffd177d241a71684006b2ca9ad19357a675df
