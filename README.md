# bot-mspacman

Trying to build the best bot to play miss pacman on atari 2600.

[Results](#results)  
[Setup](#setup)  
&nbsp;&nbsp;&nbsp;[Get the sources](#get-sources)  
&nbsp;&nbsp;&nbsp;[Install and configure Python](#setup-python)
&nbsp;&nbsp;&nbsp;[Install atari_py roms](#setup-atari-py-roms)

<a id="results"></a>

## Results

| Methods       | Average Score (50 episodes) |
| ------------- | --------------------------- |
| Random policy | 218                         |
| ...           | ...                         |

<a id="setup"></a>

## Setup

<a id="setup"></a>

### Get the sources

You may

- clone this github repository
- or download a zip containing the latest version or a given release of the code

<a id="setup-python"></a>

### Install and configure Python

1.  Install [Python 3.7 or later](https://www.python.org/downloads/).
2.  From the root folder of the project, type
    ```
    pip install virtualenv
    ```
3.  Then type
    ```
    virtualenv --python="C:\Python39\python.exe" venv
    ```
    This will create a venv subfolder.  
    Note that the path to the python.exe (v3.7) may vary on your machine.
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
