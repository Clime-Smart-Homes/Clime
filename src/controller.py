from utils import networking_utils
from app.app import process_images

conn = networking_utils.C4Network()

for value in process_images():
    print(value)
    #conn.send_value(value)
