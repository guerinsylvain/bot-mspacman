# bot-mspacman

Trying to build the best bot to play miss pacman on atari 2600.

[Overview](#overview)
[References](#references)  
[Results](#results)  
[What is LUA](#lua)  
[Bizhawk](#bizhawk)  
[Setup](#setup)  
&nbsp;&nbsp;&nbsp;[Get the sources](#get-sources)  
&nbsp;&nbsp;&nbsp;[Install and configure Bizhawk](#setup-bizhawk)  
&nbsp;&nbsp;&nbsp;[Install the CUDA Toolkit 10.1](#setup-cudatoolkit)  
&nbsp;&nbsp;&nbsp;[Install the NVIDIA CUDA Deep Neural Network library (cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1)](#setup-cudnn)  
&nbsp;&nbsp;&nbsp;[Install and configure Python](#setup-python)  
[How to train the agents](#how-to-train)  
&nbsp;&nbsp;&nbsp;[Start the agent (python)](#how-to-train-python)  
&nbsp;&nbsp;&nbsp;[Launch the ROM in the emulator and start the LUA script](#how-to-train-lua)

<a id="overview"></a>

## Overview

I started this project because I had no time to play games anymore.  
Its ultimate goal is to set up an agent that will be able to get the highest score when playing Ms PacMan on the Atari 2600 at my place.  
Here are the main parts of the chosen technical solution (tested on windows 10 64 bits):

- an emulator (Bizhawk) that will run the game
- a LUA script that will
  - init a socket client
  - grab, preprocess and send observations
  - take actions
  - control the emulator
- a python script with socket server and different learning reinforcement methods will
  - receive observations
  - train the models (dynamic programming, deep neural networks, evolutive neural network with genetic algorithms...)
  - return action(s)

<a id="references"></a>

## References

Here are some references that helped me to build this project:

- [Reinforcement Learning Explained (edx.org)](https://www.edx.org/course/reinforcement-learning-explained-2)
- [Reinforcement Learning - Introducing Goal Oriented Intelligence (Deeplizard)](https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv)

<a id="results"></a>

## Results

| Methods       | Average Score (50 episodes) |
| ------------- | --------------------------- |
| Random policy | 218                         |
| ...           | ...                         |

<a id="lua"></a>

## What is LUA

[Lua](https://www.lua.org/about.html) is a powerful, efficient, lightweight, embeddable scripting language.  
It supports procedural programming, object-oriented programming, functional programming, data-driven programming, and data description.

<a id="bizhawk"></a>

## Bizhawk

[BizHawk](https://github.com/TASVideos/BizHawk) is a multi-system emulator written in C#.  
BizHawk provides nice features for casual gamers such as full screen, and joypad support in addition to full rerecording and debugging tools for all system cores.  
LUA Functions available inside Bizhawk are documented [here](http://tasvideos.org/Bizhawk/LuaFunctions.html).

<a id="setup"></a>

## Setup

<a id="get-sources"></a>

### Get the sources

You may

- clone this github repository
- or download a zip containing the latest version or a given release of the code

<a id="setup-bizhawk"></a>

### Install and configure Bizhawk

1.  Run the PowerShell script bizhawk.ps1 located in the folder "bizhawk".  
    To install it, right-click it and select "Run with PowerShell".  
    This will download & install a fresh copy of BizHawk with all the required files in their correct locations.  
    Special thanks to [TestRunnerSRL](https://github.com/TestRunnerSRL) for this script !

    Start Bizhawk and go to Config -> Customize... -> Advanced and set Lua Core to Lua+LuaInterface.  
    NLua does not support LuaSockets properly.  
    After changing this setting, you need to close and restart the emulator for the setting to properly update.

2.  Download the ROM of the Ms Pac-Man game from this [link](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).  
    Create a new folder "ROMS" below the "BizHaw-2.3".  
    Copy the file "Ms. Pac-Man (CCE).bin" in it.

<a id="setup-cudatoolkit"></a>

### Install the CUDA Toolkit 10.1

1. Please install from the following [link](https://developer.nvidia.com/cuda-10.1-download-archive-update2)  
   Note that you can uncheck "Install Visual Studio Extensions" in the options.
   <a id="setup-cudnn"></a>

### Install the [NVIDIA CUDA Deep Neural Network library (cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1)](https://developer.nvidia.com/cudnn)

1. Follow the instructions detailed [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)  
   Make sure to install the version v7.6.5 (November 5th, 2019), for CUDA 10.1

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
    `pip install -r requirements.txt`
    <a id="how-to-train"></a>

## How to train the agents

The following steps have to be done in the described sequence:
<a id="how-to-train-python"></a>

### Start the agent (python)

1. From the root folder of the project, activate the virtual environment by typing:
   ```
   .\venv\Scripts\activate.bat
   ```
2. Launch the training script:
   `python train.py`
   <a id="how-to-train-lua"></a>

### Launch the ROM in the emulator and start the LUA script

1.  Start BizHawk
2.  In the "File" menu, chose "Open ROM".  
    Select the file "Ms. Pac-Man (CCE).bin" that you have previously copied in the "ROMS" folder below "BizHawk-2.3".
3.  Select "Lua Console" in the "Tools" menu.  
    The "Lua Console" window opens.
4.  In the "Script" menu of the "Lua Console" window, select "Open Script...".  
    Select the file "train.lua" located in the bizhawk folder.

The training should start.
If anything goes wrong, please review the [Setup section](#setup)
