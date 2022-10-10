from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light
import asyncio
import json
import socket
port = 50000
from time import sleep

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
light = C4Light(director, 31)

s=socket.socket()
host = socket.gethostname()
s.bind((host,port))
s.listen(5)
print("Server listening...")

while True:
    conn, addr = s.accept()
    print("Connection: " + repr(addr))
    with conn:
        while True:
            data = conn.recv(1024)
            if not data: 
                break
            value = int.from_bytes(data, 'big')
            print("Server received: " + str(value))
            asyncio.run(light.rampToLevel(value, 100))
            # print(asyncio.run(light.getState()))
