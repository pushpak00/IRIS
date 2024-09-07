# Iris Flower Classification using Machine Learning

## Overview

This project demonstrates the application of machine learning algorithms to classify iris flower species based on their sepal and petal measurements. The Iris dataset, which is widely used in pattern recognition literature, contains 150 observations of iris flowers with four features: sepal length, sepal width, petal length, and petal width. The goal is to classify the flowers into three species: Setosa, Versicolor, and Virginica.

## Project Structure

- `data/`: Contains the dataset used in this project.
  - `iris.csv`: The Iris dataset.
- `notebooks/`: Jupyter notebooks with exploratory data analysis (EDA) and model training.
  - `EDA_and_Model_Training.ipynb`: Notebook containing data visualization, feature engineering, and model training.
- `models/`: Trained machine learning models.
  - `iris_model.pkl`: Saved model for future predictions.
- `src/`: Source code for data preprocessing, model training, and evaluation.
  - `data_preprocessing.py`: Script for loading and preprocessing the data.
  - `model_training.py`: Script for training and saving the model.
  - `model_evaluation.py`: Script for evaluating model performance.

## Requirements

To run this project, you will need the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Project Steps

### 1. Data Exploration and Visualization

The first step involves exploring the dataset to understand the distribution of features and the relationships between them. Various plots, such as scatter plots, histograms, and box plots, are used to visualize the data.

### 2. Data Preprocessing

The dataset is cleaned and preprocessed to make it suitable for training machine learning models. This step includes handling missing values (if any), encoding categorical variables, and splitting the data into training and testing sets.

### 3. Model Training

Multiple machine learning algorithms are applied to the dataset to classify the iris species. The models used include:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

Each model's performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

### 4. Model Evaluation

The trained models are evaluated on the test set to determine their performance. The model with the highest accuracy is selected as the final model. Cross-validation is also used to ensure the model's robustness.

### 5. Model Deployment (Optional)

The selected model can be saved and deployed for future use. A simple API can be built using Flask or FastAPI to serve the model and make predictions on new data.

## Results

The project achieves a classification accuracy of over 95% on the test set using the Random Forest Classifier. The model's performance is consistent across different metrics, indicating its reliability in classifying iris species.

## Conclusion

This project successfully demonstrates how to apply machine learning algorithms to classify iris flowers. The simplicity of the Iris dataset makes it an excellent starting point for learning about classification problems in machine learning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Iris dataset is a classic dataset in the field of machine learning and is publicly available through the UCI Machine Learning Repository.
- Thanks to Scikit-learn for providing easy-to-use machine learning tools.
