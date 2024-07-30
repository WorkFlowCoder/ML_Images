import json
import torch

def readParameter():
    with open('config.json', 'r') as file:
        config = json.load(file)
    #learning_rate = config["learning_rate"]
    #batch_size = config["batch_size"]
    #epochs = config["epochs"]
    #data = config["data"]
    return config

def saveModel(model):
    torch.save(model.state_dict(),"data/model.pth")
    print("Save pytorch model state to model.pth")
