# SOS-bilinear-alternation
This repository aim to store my implementation of the SOS bilinear alternation for a torque limited simple pendulum. In particular, it should be useful to fix the bug that I still have to solve: the estimation is succeeding in providing me a meaningful Funnel but the verification gives me some fails, teoretically I am not expecting such fails since I am implementing an inner estimation of the RoA.

## Software installation #
In order to use this repository the user should clone the repository in a local folder. Then, inside the project folder, run "pip install ." in order to install the needed packages.

```
Warning: this procedure should be done in a virtual environment. I used "pipenv" for example and hence, after creating the environment with "pipenv shell", the command to run will be "pipenv install .".
```