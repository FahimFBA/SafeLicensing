# SafeLicensing

This project demonstrates a pipeline for detecting license plates in images using YOLOv8 and encrypting the detected regions with a Chaotic Logistic Map encryption algorithm. It provides a user-friendly interface built with Streamlit.

>[!TIP]
> You can directly test the application on the web using the following link: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/fahimfba/safelicensing/main/app.py)

## Features

- **License Plate Detection**: Uses the YOLOv8 model to detect license plates in uploaded images.
- **Chaotic Encryption**: Encrypts the detected license plate regions using a two-layer XOR-based chaotic logistic map algorithm.
- **Streamlit Web App**: A simple interface to upload images, detect license plates, encrypt them, and download the results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FahimFBA/SafeLicensing.git
   cd SafeLicensing
   ```

2. Install `ffmpeg` for video processing (Linux):
   ```bash
   sudo apt-get install ffmpeg
   ```

   or, for macOS:
   ```bash
   brew install ffmpeg
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 weights file (`best.pt`) and place it in the root directory of the project. You can train your own model or use a pre-trained one. This repository already have our model from [SEncrypt](https://github.com/IsratIJK/SEncrypt) located in [best.pt](./best.pt) file.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser (typically at `http://localhost:8501`).

3. Follow the steps:
   - Upload an image or provide a URL.
   - Adjust the encryption key seed using the slider.
   - Click the "Detect & Encrypt" button to process the image.

4. Download the encrypted image directly from the app.

## Docker Usage

You can run this application using Docker. There are two options: CPU-only and GPU-enabled.

### CPU Version

1. Build the Docker image:
   ```bash
   docker build -t safelicensing .
   ```

2. Run the Docker container:
   ```bash
   docker run -d -p 8501:8501 --name safelicensing_container safelicensing
   ```

### GPU Version (for NVIDIA GPU users)

1. Ensure you have the NVIDIA Container Toolkit installed. If not, follow the [official installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

2. Build the GPU-enabled Docker image:
   ```bash
   docker build -f Dockerfile.gpu -t safelicensing-gpu .
   ```

3. Run the GPU-enabled Docker container:
   ```bash
   docker run -d -p 8501:8501 --gpus all --name safelicensing_gpu_container safelicensing-gpu
   ```

### Common Steps

4. Check if the container is running:
   ```bash
   docker ps
   ```
   You should see a container named `safelicensing_container` or `safelicensing_gpu_container` in the list.

5. View the container logs:
   ```bash
   docker logs safelicensing_container  # or safelicensing_gpu_container for GPU version
   ```
   This will show you the Streamlit startup logs and any errors if they occur.

6. Open the app in your browser by navigating to `http://localhost:8501`.

If you encounter any issues:
- Ensure that port 8501 is not being used by another application.
- Check the container logs for any error messages.
- If needed, you can stop and remove the container using:
  ```bash
  docker stop safelicensing_container  # or safelicensing_gpu_container
  docker rm safelicensing_container  # or safelicensing_gpu_container
  ```
  Then, try running the container again.

Note: The GPU version requires an NVIDIA GPU and proper drivers. If you don't have a compatible GPU, use the CPU version instead.

## Workflow

1. **License Plate Detection**:
   - The YOLOv8 model is used to detect license plates in the input image. The model has been taken from [SEncrypt](https://github.com/IsratIJK/SEncrypt).
   - Detected regions are highlighted with bounding boxes.

2. **Chaotic Logistic Map Encryption**:
   - A chaotic logistic map generates two XOR-based encryption keys.
   - Pixels in the license plate regions are shuffled and encrypted in two stages.
   - The encrypted region replaces the original plate in the image.

3. **Visualization and Download**:
   - The original, detected, and encrypted images are displayed in the app.
   - Encrypted images can be downloaded as PNG files.

## Files

- `app.py`: The main Streamlit app file.
- `requirements.txt`: Python dependencies for the project.
- `best.pt`: YOLOv8 weights file (not included, add your own).

## Key Parameters

- **Encryption Key Seed**: A slider in the app adjusts the seed value for the chaotic logistic map, affecting the encryption's randomness.

## Example Screenshots

### Original Image

![Original Image](./img/lpr-tesla-license-plate-recognition-1910x1000.jpg)

### Encrypted Image

![Encrypted Image](./img/encrypted_plate.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any queries, feel free to reach out:

- **Author**: Md. Fahim Bin Amin
- **GitHub**: [FahimFBA](https://github.com/FahimFBA)

- **Other Authors**: [Rafid Mehda](https://github.com/rafid29mehda), [Israt Jahan Khan](https://github.com/IsratIJK)
