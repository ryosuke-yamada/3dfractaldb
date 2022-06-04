#! /bin/bash

variance_threshold=0.05
numof_category=10
numof_instance=10
numof_object=5
numof_scene=10

# Parameter search
python code/category_search.py --variance=${variance_threshold} --numof_classes=${numof_category}

# Generate 3D fractal model
python code/fractal_noise_mix.py --numof_classes=${numof_category} --numof_instance=${numof_instance} \
                                 --visualize

# Generate 3D fractal scene
python code/generate_scene.py --numof_classes=${numof_category} --numof_instance=${numof_instance} \
                              --numof_object=${numof_object} --numof_scene=${numof_scene}