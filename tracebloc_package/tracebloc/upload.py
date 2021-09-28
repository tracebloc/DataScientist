
import requests, json, pickle
from importlib.machinery import SourceFileLoader

class Model():

    '''
    Make sure model file and weights are in current directory
    Parameters: modelname, token

    modelname: model file name eg: vggnet, if file name is vggnet.py
    token: token recieved from login

    *******
    Call getNewModelName method on Upload to get uploded model unique name
    '''

    def __init__(self,modelname,token):
        self.__modelname = modelname
        self.__token = token
        self.__url = 'https://xray-backend-develop.azurewebsites.net/upload/'
        # self.__url = 'http://127.0.0.1:8000/upload/'
        self.__recievedModelname = self.upload()

    def getNewModelId(self):
        print(self.__recievedModelname)
        return self.__recievedModelname

    def check(self):
        try:

            #Load weights to check if it works
            w = open(f'{self.__modelname}_weights.pkl', 'rb')
            we = pickle.load(w)
            model = SourceFileLoader(self.__modelname, f'{self.__modelname}.py').load_module()
            model = model.MyModel()     
            model.set_weights(we)
            return True
        except ValueError:
            return False


    def upload(self):

        #call check function before calling upload API
        s = self.check()

        if s:

            header = {'Authorization' : f"Token {self.__token}"}
            files = {'upload_file': open(f'{self.__modelname}.py','rb'),
            'upload_weights': open(f'{self.__modelname}_weights.pkl','rb')}
            values = {"model_name": self.__modelname}
            r = requests.post(self.__url, headers = header, files=files, data=values)
            
            if r.status_code == 202:
                body_unicode = r.content.decode('utf-8')
                content = json.loads(body_unicode)
                print("Upload successful.")
                print("\n")
                return content['model_name']
            else:
                print("Upload failed. \nPlease check naming convention of model and weight files and try again.")
                print("\n")
        else:
            print("Provide correct weights and model")
