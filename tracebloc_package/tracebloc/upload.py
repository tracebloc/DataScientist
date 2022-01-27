
import requests, json, pickle
from importlib.machinery import SourceFileLoader
from termcolor import colored
import os

class Model():

    '''
    Make sure model file and weights are in current directory
    Parameters: modelname

    modelname: model file name eg: vggnet, if file name is vggnet.py

    '''

    def __init__(self,modelname,token,weights=False):
        self.__modelname = modelname
        self.__token = token
        self.weights = weights
        self.__url = 'https://xray-backend-develop.azurewebsites.net/upload/'
        # self.__url = 'http://127.0.0.1:8000/upload/'
        self.__recievedModelname = self.upload()

    def getNewModelId(self):
        # print(self.__recievedModelname)
        return self.__recievedModelname

    def checkModel(self):
        # load model from current directory
        try:
            modelFile = open(f'{self.__modelname}.py','rb')
            modelFile.close()
            if self.weights == True:
               w = self.checkWeights()
               return w
            return True
        except FileNotFoundError:
            text = colored('Model upload failed!', 'red')
            print(text,"\n")
            print(f"There is no model with the name {self.__modelname} in your folder {os.getcwd()}\n")
            print(f"Your model should be of a python file: {self.__modelname}.py")
            return False

    def checkWeights(self):
        # load model weights from current directory
        try:
            weightsFile = open(f'{self.__modelname}_weights.pkl','rb')
            weightsFile.close()
        except FileNotFoundError:
            text = colored('Model upload failed!', 'red')
            print(text,"\n")
            print(f"The model weights file does not meet the convention: expected weights name: ”{self.__modelname}_weights.pkl”. Please check your model weights file name")
            return False
        # Load weights to check if it works
        try:
            w = open(f'{self.__modelname}_weights.pkl', 'rb')
            we = pickle.load(w)
            model = SourceFileLoader(self.__modelname, f'{self.__modelname}.py').load_module()
            model = model.MyModel()     
            model.set_weights(we)
            w.close()
            return True
        except ValueError:
            text = colored('Model upload failed!', 'red')
            print(text,"\n")
            print("Provide weights compatible with provided model!")
            return False


    def upload(self):
        m = self.checkModel()
        if m:
            if self.weights == True:
                modelFile = open(f'{self.__modelname}.py','rb')
                weightsFile = open(f'{self.__modelname}_weights.pkl','rb')
                files = {'upload_file': modelFile,
                'upload_weights': weightsFile}
                values = {"model_name": self.__modelname,"setWeights": True}
            else:
                modelFile = open(f'{self.__modelname}.py','rb')
                files = {'upload_file': modelFile}
                values = {"model_name": self.__modelname,"setWeights": False}
            # upload on the server
            header = {'Authorization' : f"Token {self.__token}"}
            r = requests.post(self.__url, headers = header, files=files, data=values)
            if r.status_code == 202:
                body_unicode = r.content.decode('utf-8')
                content = json.loads(body_unicode)
                text = colored("Upload successful.", "green")
                print(text,"\n")
                return content['model_name']
            else:
                text = colored('Server: Model upload failed!', 'red')
                print(text,"\n")
                
