 # Astrophysics-x-ML
 ****

# PROJECT1
## Astronomical Object Classification

This project aims to classify astronomical objects into different categories such as stars, galaxies, and quasars using machine learning techniques. We use a dataset containing features like absolute magnitude, redshift, surface brightness, stellar mass, and age to train a Random Forest classifier.

### Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Setup](#setup)
- [Code Overview](#code-overview)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

### Introduction

Astronomical object classification is a fundamental problem in astrophysics, which helps in understanding the universe. Machine learning models can automatically categorize objects based on their characteristics, making this process more efficient and less reliant on human experts.

### Dataset

The dataset used in this project includes the following features:
- **Absolute Magnitude**: A measure of the intrinsic brightness of an astronomical object.
- **Redshift**: A measure of how much the wavelength of the object's light has been stretched due to the expansion of the universe.
- **Surface Brightness**: The brightness of the object per unit area.
- **Stellar Mass**: The mass of the stars in the object.
- **Age**: The age of the object.

The target variable, `class`, represents the category of the astronomical object, with possible values:
- 0: Star
- 1: Galaxy
- 2: Quasar

The dataset contains 1 million samples, generated synthetically for this project.

### Setup

To run this project, you need to have Python installed along with the necessary libraries. You can set up your environment as follows:

1. **Install Python**: Ensure that Python 3.7 or higher is installed on your machine.

2. **Install Dependencies**: Install the required Python libraries using pip:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib

****

# PROJECT 2
## Black Holes and Quasars Classification with Machine Learning

### Overview

This project involves generating synthetic data for black holes and quasars, preprocessing the data, training a machine learning model to classify the objects, and evaluating the model's performance. The project aims to achieve an accuracy of over 95% using a RandomForestClassifier.

### Project Structure

1. **Data Generation**: Create a synthetic dataset with 1 million entries.
2. **Data Preprocessing**: Clean and prepare the data for analysis.
3. **Exploratory Data Analysis (EDA)**: Visualize the data to understand distributions and relationships.
4. **Feature Engineering**: Create meaningful features for the model.
5. **Model Selection and Training**: Train a RandomForestClassifier.
6. **Evaluation**: Evaluate the model's performance using various metrics.
7. **Visualization**: Visualize the results for better interpretation.

### Dependencies

The following Python libraries are required:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
