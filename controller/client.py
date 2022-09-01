import socket                   # Import socket module

s = socket.socket()             # Create a socket object
host = socket.gethostname()     # Get local machine name
port = 50000                    # Reserve a port for your service.

s.connect((host, port))
while True:
    try:
        integer_val = int(input("Set light level: "))
    except:
        break

    bytes_val = integer_val.to_bytes(2, "big")

    s.send(bytes_val)

s.close()

