# Source code

In order to run the program, ensure you have the Control4 Controller and Jetson on the same network with correct IP addresses set in `controller.py`. Then, install all the required packages for Python 3.7.5 using:
`python3 -m pip install -r requirements_full.txt`

If you want to have more flexibility in your package versioning, use `requirements.txt` instead. Note that Python 3.7 or greater is required.

Once that is done, run `python3 controller.py`
## Sections
- [*app*](https://github.com/Clime-Smart-Homes/Clime/tree/main/src/app) - Application that runs different machine learning model and sends their values to the `controller`.
- [*client*](https://github.com/Clime-Smart-Homes/Clime/tree/main/src/client) - Simple website to send POST requests to `controller` to change the active model seamlesly.
- [*images*](https://github.com/Clime-Smart-Homes/Clime/tree/main/src/images) - Image storage location if you want to save images while the program is running.
- [*model*](https://github.com/Clime-Smart-Homes/Clime/tree/main/src/model) - Contains age and emotion detection models as well as their interface for the app.
- [*utils*](https://github.com/Clime-Smart-Homes/Clime/tree/main/src/utils) - Networking and drawing utilities.
