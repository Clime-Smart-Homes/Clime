import asyncio
from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light


class C4Network:
    def __init__(self):
        """Put Control4 account username and password here"""
        username = ""
        password = ""

        ip = "192.168.0.13"

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

    def send_value(self, value):
        asyncio.run(self.light.rampToLevel(value, 100))
