import asyncio
from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light
from decouple import config


class C4Network:
    def __init__(self, ip):
        """Put Control4 account username and password here"""
        username, password = self.get_account_info()

        if username is None or password is None:
            raise Exception("Username or password is missing from .env file in src directory.")

        """Authenticate with Control4 account"""
        account = C4Account(username, password)
        asyncio.run(account.getAccountBearerToken())

        """Get and print controller name"""
        accountControllers = asyncio.run(account.getAccountControllers())
        print(accountControllers["controllerCommonName"])

        """Get bearer token to communicate with controller locally"""
        director_bearer_token = asyncio.run(
                    account.getDirectorBearerToken(accountControllers["controllerCommonName"])
                    )["token"]

        """Create new C4Director instance"""
        director = C4Director(ip, director_bearer_token)


        """Create new C4Light instance, put your own device for the ID"""
        self.light = C4Light(director, 31)

        if self.get_director_status(self.light) is False:
            raise Exception("Connection to director could not be established.")

    def send_value(self, value):
        asyncio.run(self.light.rampToLevel(value, 100))

    def get_account_info(self):
        try:
            username = config('C4_USERNAME')
            password = config('C4_PASSWORD')
        except:
            return None, None

        return (username, password)

    def get_director_status(self, light):
        #try:
        asyncio.run(light.getState())
        return True
        #except:
        #    return False
            
