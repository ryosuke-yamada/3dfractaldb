# Point Cloud Fractal DataBase (PC-FractalDB) 

Run the python script ```PCFractalDB_render.sh```, you can get our PC-FractalDB.

## Requirements

* Python 3 (worked at 3.7)

* open3D (worked at 0.13.0)

* numpy (worked at 1.19.0)

* jsons

## Running the code

We prepared execution file PCFractalDB_render.sh in the top directory. The execution file contains our recommended parameters. Please type the following commands on your environment. You can execute the Fractal Category Search, FractalNoiseMix, 3D Fractal Scene Generate, PC-FractalDB Construction.

```bash PCFractalDB_render.sh```

The folder structure is constructed as follows.

```misc
./
  PC-FractalDB/
    3DIFS_param/
      000000.csv
      000001.csv
      ...
    3Dfractalmodel/
      000000/
        000000_0000.ply
        000000_0001.ply
        ...
      ...
    3Dfractalscene/
      scene_00000.json
      scene_00001.json
      ...
    ...
  render.sh
```

## Training / validation in 3D object detection

 We employed spatiotemporal 3D Convolutional Neural Networks (CNN). We mainly used ```PointContrast``` and ```VoteNet``` for main experiments and ```ContrastiveSceneContexts``` for additional experiments.

* [PointContrast](https://github.com/facebookresearch/PointContrast)
* [VoteNet](https://github.com/facebookresearch/votenet)
* [ContrastiveSceneContexts](https://github.com/facebookresearch/ContrastiveSceneContexts)
