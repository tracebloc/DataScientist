#import useful libraries

import requests
import json
import getpass, os
import ast
import uuid

class User():

    '''
    Parameters: username, password

    ***
    Please provide a valid username and password
    Call getToken method on Login to get new token for provided
    username and password
    '''

    def __init__(self,username='',password=''):
        self.__username = username
        self.__password = password
        self.__token = self.login()

    def getToken(self):

        if len(self.__token) == 0:
            print("Please login and try again.")
        else:
            print("Token received.")
            return self.__token

    def login(self):
        '''Function to get Token for username provided'''
        try:
            url = "https://xray-backend.azurewebsites.net/api-token-auth/"
            r = requests.post(url, data = {"username": self.__username, "password": self.__password})
            if r.status_code == 200:
                print("Login successful.")
                print("\n")
            token = json.loads(r.text)['token']
        
        except Exception as e:
            
            print("Login credentials are not correct. Please try again.")
            return ""
        return token