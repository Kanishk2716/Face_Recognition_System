# Face_Recognition_System

## Installation

1. Clone the repository or download the ZIP file.

   ```sh
   git clone https://github.com/your-username/face-recognition-system.git
   cd face-recognition-system
   ```

2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Using Your Own Dataset

If you want to create your own dataset, follow these steps:

1. Run the `Dataset.py` file to collect and store face images:
   
   ```sh
   python Dataset.py
   ```

2. Train the model using the collected dataset:
   
   ```sh
   python Training.py
   ```

3. Run the detection script to recognize faces:
   
   ```sh
   python Detection.py
   ```

### Using the Provided Dataset

If you want to use the pre-existing dataset (Ronaldo & Messi images), you can directly run the detection script:

```sh
python Detection.py
```

Sample test images are available inside:

```
dataset/test_images/
```


