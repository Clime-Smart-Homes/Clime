from utils import networking_utils
from app.app import process_images


def main():
    conn = networking_utils.C4Network()

    for value in process_images():
        print("Sending: " + str(value))
        conn.send_value(value)


if __name__ == '__main__':
    main()
