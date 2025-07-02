# 🌸 Iris Flower Classification with PyTorch (Manual Neural Network)

This project demonstrates how to build a two-layer neural network from scratch (without high-level PyTorch wrappers) to classify Iris flowers into three species based on their features.

## 📁 Dataset

We use the classic **Iris dataset**, which contains 150 samples of flowers belonging to 3 species:

- **Setosa**
- **Versicolor**
- **Virginica**

Each sample has 4 features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dataset should be in a CSV file named `iris.csv` with the following headers:


## 📊 Project Workflow

1. **Load and preprocess the dataset**
   - Label encode the species
   - Normalize the features
   - Split into 80% training and 20% testing data

2. **Manual Model Definition**
   - A fully-connected 2-layer neural network
   - No use of `nn.Module` or `nn.Sequential`

3. **Training**
   - Manual forward pass
   - CrossEntropy loss calculation
   - Backward pass using `autograd`
   - Parameter updates using SGD

4. **Evaluation**
   - Accuracy calculated on training and test data
   - Plot of accuracy over 50 epochs

## 🧠 Neural Network Architecture

- **Input Layer**: 4 neurons (features)
- **Hidden Layer**: 16 neurons with ReLU activation
- **Output Layer**: 3 neurons (classes)

## 🛠 Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

Install all dependencies using pip:

```bash
pip install torch pandas numpy scikit-learn matplotlib
