import torch
import torch.utils.data as DataSet
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import onnxruntime
import torch.onnx as onnx
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class SimpleNeuralNetwork(nn.Module):
    FashionMNISTLabel = {0: "T-shirt",1: "Trouser",2: "Pullover",3: "Dress",4: "Coat",5: "Sandal",6: "Shirt",7: "Sneaker",8: "Bag",9: "Ankle Boot"}
    MNISTLabel = {0: "0",1: "1",2: "2",3: "3",4: "4",5: "5",6: "6",7: "7",8: "8",9: "9"}
    CIFARLabel = {0:"airplane",1: "automobile",2: "bird",3: "cat",4: "deer",5: "dog",6: "frog",7: "horse",8: "ship",9: "truck"}
    STLLabel = {0:"airplane",1: "bird",2: "car",3: "cat",4: "deer",5: "dog",6: "horse",7: "monkey",8: "ship",9: "truck"}

    def __init__(self,config):
        super(SimpleNeuralNetwork,self).__init__()
        size = 28*28
        nbLabel = 10
        if config["data"]=="CIFAR":
            size = 32*32*3
        elif config["data"]=="STL":
            size = 96*96*3
        self.flatten = nn.Flatten()
        modules = []
        nbLayers = 2
        if "nbLayers" in config:
            nbLayers = config["nbLayers"]
        for i in range(0,nbLayers):
            modules.append(nn.Linear(size,size))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(size,nbLabel))
        modules.append(nn.ReLU())
        self.linear_relu_stack = nn.Sequential(*modules)
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    def generateTrainTest(self,config):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        transformVoc = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        if(config["data"]=="FashionMNIST"):
            labelName = self.FashionMNISTLabel
            trainingData = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
            test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())
        elif(config["data"]=="MNIST"):
            labelName = self.MNISTLabel
            trainingData = datasets.MNIST(root="data",train=True,download=True,transform=ToTensor())
            test_data = datasets.MNIST(root="data",train=False,download=True,transform=ToTensor())
        elif(config["data"]=="CIFAR"):
            labelName = self.CIFARLabel
            trainingData = datasets.CIFAR10(root="data", train=True,download=True, transform = transform)
            test_data = datasets.CIFAR10(root="data", train=False,download=True, transform = transform)
        elif(config["data"]=="STL"):
            labelName=self.STLLabel
            trainingData = datasets.STL10(root="data", split='train',download=True, transform = transform)
            test_data = datasets.STL10(root="data", split='test',download=True, transform = transform)
        elif(config["data"]=="PASCAL_VOC"):
            labelName=self.VOCLabel
            trainingData = datasets.VOCDetection(root="data",year='2007',image_set='train',download=True,transform=transformVoc)
            test_data = datasets.VOCDetection(root="data",year='2007',image_set='val',download=True,transform=transformVoc)
        return labelName, trainingData, test_data
    
    def run(self,config,trainingData,test_data):
        trainLoader = DataLoader(trainingData,batch_size=64,shuffle=True)
        testLoader = DataLoader(test_data,batch_size=64,shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(),lr=config["learning_rate"])
        for t in range(config["epochs"]):
            print(f"Epoch {t+1}\n --------------------------------------")
            self.train_loop(trainLoader, self, loss_fn,optimizer)
            self.test_loop(testLoader,self,loss_fn)
        print("Done !")
    
    def train_loop(self,dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (x,y) in enumerate(dataloader):
                #Compute prediction and loss
                pred = model(x)
                loss = loss_fn(pred,y)
                #Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch%100 == 0:
                    loss, current = loss.item(), batch * len(x)
                    print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test_loop(self,dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in dataloader:
                pred = model(x)
                test_loss += loss_fn(pred,y).item()
                correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def predict(self, test_data, labelName):
        input_image = torch.zeros((1,28,28))
        onnx_model = 'data/model.onnx'
        onnx.export(self,input_image,onnx_model)
        session = onnxruntime.InferenceSession(onnx_model,None)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        countGoodPredict = 0
        for i in range(0,len(test_data)):
            x, y = test_data[i][0], test_data[i][1] # for all
            result = session.run([output_name],{input_name: x.numpy()})
            predicted, actual = labelName[result[0][0].argmax(0)], labelName[y]
            if predicted == actual:
                countGoodPredict+=1
        print("Good prediction :",countGoodPredict,"/",len(test_data),'(',(countGoodPredict/len(test_data)),'%)') ########################### Amelioration de l'affichage
    
    def predictForUniqueItem(self, test_data, labelName, pos):
        input_image = torch.zeros((1,28,28))
        onnx_model = 'data/model.onnx'
        onnx.export(self,input_image,onnx_model)
        x, y = test_data[pos][0], test_data[pos][1] # unique item
        session = onnxruntime.InferenceSession(onnx_model,None)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        result = session.run([output_name],{input_name: x.numpy()})
        predicted, actual = labelName[result[0][0].argmax(0)], labelName[y]
        print(f'Predicted: : "{predicted}", Actual : "{actual}"')
    
    def showImage(self,data,labelName,pos,config):
        trainLoader = DataLoader(data,batch_size=64,shuffle=False)
        train, label_train = next(iter(trainLoader))
        imgZero = train[pos].squeeze()
        labelZero = label_train[pos]
        label_here = list(labelName.values())[labelZero]
        if config["data"] in {"CIFAR","STR"}:
            plt.imshow(np.transpose(imgZero,(1,2,0)))
        else:
            plt.imshow(imgZero,cmap="gray")
        plt.title("Image with label : "+str(label_here))
        plt.show()