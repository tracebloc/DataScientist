
import requests, json

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
#        self.__url = 'http://127.0.0.1:8000/upload/'
        self.__url = 'https://xray-backend.azurewebsites.net/upload/'
        self.__recievedModelname = self.upload()

    def getNewModelId(self):
        print(self.__recievedModelname)
        return self.__recievedModelname


    def upload(self):

        header = {'Authorization' : f"Token {self.__token}"}
        files = {'upload_file': open(f'{self.__modelname}.py','rb'),
        'upload_weights': open(f'{self.__modelname}_weights.pkl','rb')}
        values = {"model_name": self.__modelname}
        r = requests.post(self.__url, headers = header, files=files, data=values)
        body_unicode = r.content.decode('utf-8')
        content = json.loads(body_unicode)
        if r.status_code == 202:
            print("Upload successful.")
            print("\n")
            return content['model_name']
        else:
            print("Upload failed. \nPlease check naming convention of model and weight files and try again.")
            print("\n")
