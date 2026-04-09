# 🌲 Random Forest Classification Dashboard

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![License](https://img.shields.io/badge/License-MIT-green)


[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?logo=rocket)](https://your-streamlit-app-link.streamlit.app)

</p>

---

## 🚀 Overview

An interactive **Random Forest Classification Dashboard** built with Streamlit that demonstrates how ensemble learning improves prediction performance by combining multiple decision trees.

The app visualizes **decision boundaries, model accuracy, and predictions**, helping users understand how Random Forest works in practice.

---

## ✨ Features

* 🌲 Train a Random Forest classifier
* 🎛️ Adjustable hyperparameters:

  * Number of trees (n_estimators)
  * Criterion (gini / entropy)
  * Max depth
* 📊 Accuracy & confusion matrix
* 🗺️ Decision boundary visualization
* 🔮 Real-time prediction tool
* 👀 Dataset preview

---

## 🛠️ Tech Stack

* Python 3.x
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📂 Project Structure

```bash
.
├── app.py
├── Social_Network_Ads.csv
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/random-forest-dashboard.git
cd random-forest-dashboard
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 📊 Model Details

* Algorithm: **Random Forest Classifier**
* Type: Ensemble Learning (Multiple Decision Trees)
* Features:

  * Age
  * Estimated Salary
* Target:

  * Purchase Decision (0 / 1)

---

## 📈 Visualizations

* 🗺️ Decision boundary (Training & Test sets)
* 📊 Confusion matrix
* 📈 Accuracy metric
* 👀 Dataset preview

---

## 🧠 How It Works

1. Dataset is loaded from CSV
2. Data is split into training and test sets
3. Features are scaled
4. Random Forest model is trained using multiple trees
5. Predictions are combined for final output

---

## 🔮 Live Prediction

Users can input:

* Age
* Salary

The model predicts whether the user will **purchase or not**.

---

## 📁 Dataset

The dataset (`Social_Network_Ads.csv`) includes:

* Age
* Estimated Salary
* Purchase decision (target)

---

## 🚀 Deployment

Deploy easily using **Streamlit Cloud**:

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create a new app
4. Select your repository
5. Deploy 🎉

---

## 🔮 Future Improvements

* 🌲 Feature importance visualization
* 🧠 Compare with other models (Decision Tree, SVM, Naive Bayes)
* 📊 Interactive Plotly charts
* 📉 Hyperparameter tuning visualization

---

## 👨‍💻 Author

**Chinmay V Chatradamath**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
