# Cancer-Detection
Skin Cancer Detection
This repository contains the code for a skin cancer detection project using Streamlit, EfficientNet-B4, and ResNet-101. The goal of this project is to develop a machine learning model that can accurately classify images of skin lesions, aiding in the early detection and diagnosis of skin cancer.

Dataset
The model was trained on a large dataset of skin lesion images collected from various sources, including medical databases and research institutions. The dataset, providing a diverse range of examples for the model to learn from.

Model Architecture
Two state-of-the-art convolutional neural networks (CNNs) were employed for this project:

EfficientNet-B4: EfficientNet is a family of CNN architectures that have achieved outstanding performance while maintaining efficient resource usage. EfficientNet-B4 is specifically known for its balance between model size and accuracy, making it a suitable choice for this project.

ResNet-101: ResNet is a widely used CNN architecture that introduces residual connections to address the issue of vanishing gradients. ResNet-101, with its 101 layers, has demonstrated excellent performance on various computer vision tasks.

Both models were trained using transfer learning, leveraging pre-trained weights from ImageNet. By fine-tuning these models on the skin cancer dataset, we can benefit from their feature extraction capabilities while adapting them to the specific task of skin lesion classification.

Streamlit Web App
To provide an intuitive and user-friendly interface for skin cancer detection, this project utilizes Streamlit, a Python library for building interactive web applications. The Streamlit web app allows users to upload images of skin lesions and obtain predictions from the trained models.
