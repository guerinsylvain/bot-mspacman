# bot-mspacman

Trying to build the best bot to play miss pacman on atari 2600.

[Environment](#environment)
[Results](#results)  
[Setup](#setup)  
&nbsp;&nbsp;&nbsp;[Get the sources](#get-sources)  
&nbsp;&nbsp;&nbsp;[Install the CUDA Toolkit 11.2](#setup-cudatoolkit)  
&nbsp;&nbsp;&nbsp;[Install the NVIDIA CUDA Deep Neural Network library (cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0,11.1 and 11.2)](#setup-cudnn)  
&nbsp;&nbsp;&nbsp;[Install and configure Python](#setup-python)  
&nbsp;&nbsp;&nbsp;[Install atari_py roms](#setup-atari-py-roms)

<a id="environment"></a>

## Environment

Atari 2600 MsPacman.  
Emulated through [gym openai](https://gym.openai.com/envs/MsPacman-v0/).  
The game runs at 60fps.  
Possible actions: ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']

<a id="results"></a>

## Results

| Methods                                                                                     | Average Score (100 episodes) | Parameters                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Random policy                                                                               | 545                          | frameskip=20                                                                                                                                                                          |
| double deep q learning convolution neural network with last screenshot observation as input | 3378                         | episodes=3600, frameskip=20, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01, sample_size=200, num_epochs=1, discount_rate=0.95, adam optimizer, learning rate = 0.001, huber loss |

<a id="setup"></a>

## Setup

<a id="setup"></a>

### Get the sources

You may

- clone this github repository
- or download a zip containing the latest version or a given release of the code

<a id="setup-cudatoolkit"></a>

### Install the CUDA Toolkit 11.2

1. Please install from the following [link](https://developer.nvidia.com/cuda-11.2.2-download-archive)  
   Note that you can uncheck "Install Visual Studio Extensions" in the options.
   <a id="setup-cudnn"></a>

### Install the [NVIDIA CUDA Deep Neural Network library (cuDNN cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0,11.1 and 11.2)](https://developer.nvidia.com/cudnn)

1. Follow the instructions detailed [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)  
   Make sure to install the version cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0,11.1 and 11.2

<a id="setup-python"></a>

### Install and configure Python

1.  Install [Python 3.8](https://www.python.org/downloads/).
2.  From the root folder of the project, type
    ```
    pip install virtualenv
    ```
3.  Then type
    ```
    virtualenv --python="C:\Python38\python.exe" venv
    ```
    This will create a venv subfolder.  
    Note that the path to the python.exe may vary on your machine.
4.  From the root folder of the project, activate the virtual environment by typing:
    ```
    .\venv\Scripts\activate.bat
    ```
5.  Install packages:
    ```
    pip install -r requirements.txt
    ```

<a id="setup-atari-py-roms"></a>

### Setup atari_py roms

See documentation [here](https://github.com/openai/atari-py#roms)
