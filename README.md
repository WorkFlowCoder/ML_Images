# ML_Images : Image Class Prediction

In this work we present a simple ML project to classified images.

## How to use this project

### Config file

You have a config.json file with :

```
{
    "learning_rate" : 1e-3,
    "batch_size" : 64,
    "epochs" : 10,
    "data" : "FashionMNIST",
    "training" : false,
    "modelPath" : "data/model.pth",
    "saveModel" : false,
    "nbLayers" : 2
}
```

Learning parameters :

 - "learning_rate"
 - "epochs"
 - "nbLayers"
 - "batch_size"

Choice your data :

 - "data"
 - "modelPath" (if you have already launched a model)

Choice your work :

 - "saveModel"
 - "training" (No need if you have already launched and saved your model)

### Launch this project

To run it, simply launch main.py with the command :

```
python main.py
```

### requirements

To execute the code, you need to have installed :

 - torch (pip install torch)
 - torchvision (pip install torchvision)
 - matplotlib (sudo apt-get install python-matplotlib)
 - onnxruntime (pip install onnxruntime)
 - numpy (pip install numpy)
 - json (pip install jsonlib)
 - os
