import json
import threading
from utils import networking_utils
from app.app import App
from bottle import Bottle, run, request, post, route
from socket import gethostname


def main():
    global SWITCH_MODEL 
    global NEXT_MODEL 

    SWITCH_MODEL = False

    #try:
    conn = networking_utils.C4Network("192.168.1.47")
    #except Exception as e:
    #    print("Unable to connect to Control4 Director.")
    #    print(str(e))
    #    return

    threading.Thread(target=server, daemon=True).start() # Run as daemon so thread stops when main thread stops

    app = App()

    for value in app.process_images():
        if SWITCH_MODEL:
            SWITCH_MODEL = False
            app.switch_model(NEXT_MODEL)

        # print("Sending: " + str(value))
        conn.send_value(value)

def switch_model(model_name):
    global SWITCH_MODEL
    global NEXT_MODEL

    SWITCH_MODEL = True
    NEXT_MODEL = model_name

def server():
    app = Bottle()
    @app.route('/health')
    def get_health():
        return "Health check: passed"

    @app.post('/change_model')
    def change_model():
        data = request.body.read()
        model_name = request.forms.get("model")
        if model_name is not None:
            switch_model(model_name)
            return f"Switched to {model_name} model.\n"
        return "Model name not found."

    run(app, host='192.168.1.45', port=8080)


if __name__ == '__main__':
    main()
