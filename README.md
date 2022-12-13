# Clime

Making smart homes smarter with machine learning. This project integrates machine learning algorithms on the NVIDIA Jetson with a Control4 Smart Home Controller to reduce water and electricity waste. 
Senior Capstone project for University of Utah ECE department, Fall 2022.

# Overview Diagram
![Diagram](https://github.com/Clime-Smart-Homes/Clime/blob/main/clime_algorithm.jpeg)

Three different modules are present:
- *A.I. module* - Computer running computer vision models with a camera. We used a [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) for this job.
- *Controller* - Smart home controller, here it is the [Control4 Controller](https://www.control4.com/solutions/products/controllers/).
- *Output* - Actuator that changes something in the home. In our case, it was a [smart faucet](https://github.com/Clime-Smart-Homes/Clime/tree/main/pcb/faucet/faucet) or standard light switch.

## Project hierarchy:
- [*src*](src) - All source code for connecting to Control4 director and running computer vision models.
- [*pcb*](pcb) - Diagrams and design documents for PCB built to convert Control4 outlet dimmer to 0-10V DC for faucet motors.
- [*venv*](venv) - Virtual Python 3.7 environment that should make setup easier.

## Team:
- Hibban Butt
- Zach Phelan
