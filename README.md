**Malaria Detection**
Malaria Detection using Convolutional Neural Networks
Introduction
Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. In this project, we aim to develop a deep learning model to detect malaria from microscopic images of blood smears. We use Convolutional Neural Networks (CNNs) to classify the images as either infected or uninfected.

Dataset
The dataset consists of microscopic images of blood smears collected from patients infected and uninfected with malaria. Each image is labeled with the corresponding class (infected or uninfected). The dataset is divided into two main folders: "Parasitized" and "Uninfected", each containing images of the respective class.

Preprocessing
Loaded images from the "Parasitized" and "Uninfected" folders, resizing them to a fixed size of 64x64 pixels.
Assigned labels to the images based on the folder name.
Split the dataset into training and testing sets, with a test size of 20%.
Normalized pixel values to a range between 0 and 1.
Applied data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping to increase the diversity of training data.
Model Architecture
Simple CNN Model
Implemented a simple CNN model consisting of three convolutional layers with max-pooling, followed by a dense layer with ReLU activation and dropout, and a final output layer with a sigmoid activation function.
Compiled the model using the Adam optimizer and binary cross-entropy loss.
Transfer Learning Models
Utilized pre-trained CNN models (VGG16, MobileNetV2, ResNet50) as feature extractors.
Added a Global Average Pooling layer followed by a dense layer with ReLU activation and dropout, and a final output layer with a sigmoid activation function.
Compiled the models using the Adam optimizer and binary cross-entropy loss.
Training
Trained all models using the augmented data generator for 12 epochs.
Saved the best-performing model based on validation accuracy.
Evaluation
Plotted the training and validation accuracy and loss for all models.
Evaluated the models on the test set and analyzed their performance.
Results
The simple CNN model achieved an accuracy of 95% on the test set.
The transfer learning models (VGG16, MobileNetV2, ResNet50) achieved comparable or slightly lower accuracy compared to the simple CNN model.
Conclusion
In this project, we developed a deep learning model to detect malaria from microscopic images of blood smears. The simple CNN model achieved promising results, demonstrating the effectiveness of CNNs in medical image classification tasks. Transfer learning models also showed potential but require further optimization and fine-tuning for better performance.
