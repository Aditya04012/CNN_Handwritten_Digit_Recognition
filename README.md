# 🧠 CNN Handwritten Digit Recognition

This project aims to classify handwritten digits (0–9) using Convolutional Neural Networks (CNNs) trained on the MNIST dataset. Several deep learning architectures were tested and evaluated, including:

- Artificial Neural Network (ANN)
- LeNet-5
- MiniVGGNet

## 📁 Repository Structure

```
CNN_Handwritten_Digit_Recognition/
│
├── lenet_model.ipynb               # LeNet-5 architecture
├── mini_VGGNet_architecture.ipynb  # MiniVGGNet architecture
├── README.md                       # Project documentation
```

## 📊 Dataset

- **MNIST**: A dataset of handwritten digits consisting of:
  - 60,000 training images
  - 10,000 test images
- Each image is grayscale and **28x28** in size.

You can automatically load the dataset using Keras:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---

## 🧪 Architectures Implemented

### 1. Artificial Neural Network (ANN)
- Flatten layer
- Dense layers with ReLU activation
- Final layer with Softmax

### 2. LeNet-5
- Two convolutional layers
- Two pooling layers
- Fully connected dense layers
- ReLU and softmax activations
- 
📌 View full architecture here:
👉 [LeNet-5_architecture.ipynb](https://github.com/Aditya04012/CNN_Handwritten_Digit_Recognition/blob/main/LeNet_5_architecture.ipynb)

### 3. MiniVGGNet
- Multiple stacked Conv2D layers
- MaxPooling2D and Dropout layers
- Final Dense layers

📌 View full architecture here:
👉 [mini_VGGNet_architecture.ipynb](https://github.com/Aditya04012/CNN_Handwritten_Digit_Recognition/blob/main/mini_VGGNet_architecture.ipynb)

---

## 📈 Accuracy Comparison

| Model         | Accuracy (%) |
|---------------|--------------|
| ANN           | 97.59        |
| LeNet         | 98.81        |
| MiniVGGNet    | 99.23        |

---

## ⚙️ Requirements

- Python ≥ 3.7
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

Install dependencies using:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
tensorflow>=2.0
numpy
matplotlib
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Aditya04012/CNN_Handwritten_Digit_Recognition.git
cd CNN_Handwritten_Digit_Recognition
```

2. Open Jupyter Notebook:

```bash
jupyter notebook
```

3. Run the desired `.ipynb` file (e.g., `mini_VGGNet_architecture.ipynb`)

---

## 📌 Key Results

- **MiniVGGNet** outperformed all other models, achieving a **99.23%** test accuracy.
- Use of **Dropout**, and **MaxPooling** improved generalization.
- All models were evaluated on the MNIST test set.

---

## 📷 Sample Predictions

(Add a few prediction result images here under `/images` folder and embed them like below:)

```markdown
![Sample Prediction](images/sample_output_1.png)
```

---

## 🧠 Author

**Aditya Bhatnagar**

- GitHub: [Aditya04012](https://github.com/Aditya04012)
- Email: adityabhatnagar0403@gmail.com

---

## ⭐ Star the Repo

If you found this helpful, please consider ⭐ starring the repository!
