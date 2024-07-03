# Sketch Recognition Software with CNN
This is my university one-year group thesis project!
Sketch Recognition Software is an intelligent drawing assistant that helps users improve their drawing skills by providing real-time feedback on the accuracy of their sketches. This project leverages deep learning techniques to recognize and assess freehand drawings, offering valuable insights to enhance artistic expression.

## Overview

Artistic expression has been a fundamental aspect of human culture, evolving with advancements in technology. Our project explores the intersection of art and technology, aiming to provide a user-friendly software tool that leverages machine learning to recognize and evaluate sketched objects. By employing a convolutional neural network (CNN) trained on a substantial dataset, the software offers a remarkable 94% accuracy in identifying sketched objects. This work contributes to individual artistic expression and bridges the gap between technology and creativity, demonstrating the potential of deep learning in the realm of art.

![overview](https://github.com/esikaupoma/Sketch-Recognition-Software-with-CNN/assets/126023004/1478c830-98ab-4ff0-a7ba-107e4c61b966)

## Key Features

- **CNN Model**: Utilizes a CNN architecture with 162,078 trainable parameters for accurate sketch recognition.
- **Dataset**: Curated a dataset of 50,000 sketches for training and testing, ensuring diversity and relevance.
- **User Interface**: Designed a vibrant and engaging UI using Flutter, accessible across iOS, Android, and web platforms.
- **Training Optimization**: Implemented learning rate scheduling and model checkpointing for efficient training and model performance.
- **Performance Metrics**: Achieved 94% accuracy on test datasets, validated through precision, recall, and F1-score metrics.
- **Experimentation**: Explored various parameters (epochs, image sizes, convolutional layers) to optimize model accuracy and training efficiency.

## Technical Details

- **Dataset**: Trained on a dataset of 50,000 sketches from Google’s Quick Draw dataset and tested on 10,000 sketches.
- **Model**: Utilizes a convolutional neural network (CNN) for object recognition.
- **Backend**: Developed using Django REST framework to handle API requests.
- **Frontend**: Cross-platform UI developed using Flutter framework.

## Datasets

We used two primary datasets for training our model:

1. **Cybertron Sketches:** 
   - **Source:** Kaggle
   - **Description:** 250 items, each with 80 sketch images in PNG format.
   - **Initial Model Accuracy:** 42%
   - [Cybertron Sketches Dataset](https://www.kaggle.com/datasets/mukeshgurpude/cybertron-sketches)
   - Used items in our work :
     
     ![dataset1](https://github.com/esikaupoma/Sketch-Recognition-Software-with-CNN/assets/126023004/edde32b1-1b74-4027-bc90-0f1fbdc13576)

2. **QuickDraw Sketches:** 
   - **Source:** Kaggle
   - **Description:** 382 items, each with approximately 100,000 sketch images in NDJSON format.
   - **Refined Dataset:** 10 classes, each with 5,000 images in PNG format.
   - **Final Model Accuracy:** 94%
   - [QuickDraw Sketches Dataset](https://www.kaggle.com/datasets/google/tinyquickdraw)
   - Used items in out work :
     
     ![dataset2](https://github.com/esikaupoma/Sketch-Recognition-Software-with-CNN/assets/126023004/caa50ffe-d61f-4d26-a3fe-b3cd5351ccc2)

## Technologies Used

- **Deep Learning:** Convolutional Neural Networks (CNNs) for sketch recognition.
- **Framework:** Flutter for a cross-platform user interface.
- **Backend:** Python with TensorFlow/Keras for model training and evaluation.

## Installation

### Prerequisites

- Python 3.x
- TensorFlow
- Django
- Flutter

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sketch-recognition-software.git
    cd sketch-recognition-software
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the Django backend:
    ```bash
    cd backend
    python manage.py migrate
    python manage.py runserver
    ```

4. Set up the Flutter frontend:
    ```bash
    cd ../frontend
    flutter pub get
    flutter run
    ```

## Usage

1. Run the backend server:
    ```bash
    python manage.py runserver
    ```

2. Run the Flutter application:
    ```bash
    flutter run
    ```

3. Start sketching on the canvas and receive real-time feedback on your drawings.

### Model Performance

- **Accuracy**: Achieved 94% accuracy on test datasets.
- **Metrics**: Precision, recall, and F1-score metrics demonstrated robust model performance across diverse sketches.

### User Interface

- Designed an intuitive and colorful UI to engage users, particularly children, in exploring their artistic talents.
  
## Acknowledgements

- Thanks to my university for supporting this one-year group thesis project.
- Special thanks to the team members who contributed to this project.
- Thanks to the authors of the datasets and libraries used in this project.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any improvements or fixes.
