<!-- GETTING STARTED -->
# UTS AI Showcase 2024 - HerdWatch Project
### About 
Livestock management is a crucial aspect of agriculture, requiring efficient methods for counting and monitoring animal behaviour. Traditional methods rely heavily on manual labour and can be time-consuming, costly, and prone to errors. In recent years, computer vision techniques have emerged as a promising solution to automate these processes. This project aims to develop a system for livestock counting and behaviour detection using computer vision algorithms. The proposed system utilises state-of-the-art deep learning techniques to process images or video footage captured from surveillance cameras installed in livestock facilities or drones. In this system, we used YOLOv8 medium model to detect and count cattle.

### Prerequisites
* A local computer or a compatible edge device
* Python 3.9
* NVIDIA GPU (optional and use PyTorch model)
* MAC GPU (optional and use Core ML model)

### Installation

1. Go to The repository
    ```sh
    cd <path_to_folder>/uts-ai-showcase-herdwatch
    ```
2. Create a virtual environment named `herdwatchenv`
    ```sh
    python -m venv herdwatchenv
    ```
3. Activate the virtual environment `herdwatchenv`:

    Linux/MacOS:
    ```sh
    source herdwatchenv/bin/activate
    ```
    Windows:
    ```sh
    herdwatchenv\Scripts\activate
    ```
4. Once Virtual Environment is active, the terminal will look like the following:

    Linux/MacOS:
    ```sh
    (herdwatchenv) apple... $ `or` %
    ```
    Windows:
    ```sh
    (herdwatchenv) C:\.... >
    ```
5. Upgrade pip:
    ```sh
    pip install --upgrade pip
    ```
6. Install the requirements now (make sure venv is activated):
    ```sh
    pip install -r requirements.txt
    ```
    This shall install all the required libraries. If version issues occur - remove version numbers.

<!-- USAGE EXAMPLES -->
## Usage

Follow the steps to start the application after the libraries are installed:

1. Go to the directory `mysite`
    ```sh
    cd mysite
    ```
2. Run the following command (required after every major code update, and first installation)
    ```sh
    python manage.py makemigrations
    ```
3. Run the following command (required after every major code update, and first installation)
    ```sh
    python manage.py migrate 
    ```
4. Run the following command (this starts the application)
    ```sh
    python manage.py runserver
    ```