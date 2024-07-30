from model import *
from utils import *
import os

def main():
    # Lecture du fichier de paramètrage
    config = readParameter()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initalisation du modèle et des données
    try:
        model = SimpleNeuralNetwork(config).to(device)
    except RuntimeError as e:
        if 'DefaultCPUAllocator: can\'t allocate memory' in str(e):
            print("Error: Unable to allocate memory. Try reducing the model size or ensure that sufficient memory is available.")
            return
        else:
            raise
    labelName, train, test = model.generateTrainTest(config)
    if config["training"] or not(os.path.exists(config["modelPath"])):
        model.showImage(test,labelName,15,config)
        # Entrainement
        model.run(config,train,test)
        if config["saveModel"]:
            # Sauvegarde du model
            saveModel(model,"model"+config["data"]+".pth")
    else:
        # Lecture du model
        model.load_state_dict(torch.load("data/model.pth"))
        model.eval()

    # test for unique element
    model.predictForUniqueItem(test,labelName,15)
    # Show image
    model.showImage(test,labelName,15,config)
    # test for all element
    model.predict(test,labelName)

if __name__ == "__main__":
    main()