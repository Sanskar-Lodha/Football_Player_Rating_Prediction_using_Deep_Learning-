# ⚽ Football Player Rating Prediction using Deep Learning

This project builds a **deep learning model** to predict the performance **rating of soccer players** based on their match statistics.  
It uses a dataset of player stats, processes the features, and trains a **neural network** using TensorFlow/Keras to make predictions.

---

## 📌 Project Overview
- **Goal:** Predict the match **rating** of a soccer player given their performance statistics.  
- **Approach:**
  - Data preprocessing (handling missing values, encoding categorical features, feature scaling)
  - Feature correlation analysis & visualization
  - Model training using a feed-forward neural network
  - Evaluation with **MAE** and **MSE**
  - Prediction of a new player’s rating using custom stats  

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Data visualization
- **Scikit-learn** – Data preprocessing, metrics
- **TensorFlow / Keras** – Deep learning model

---

## 📂 Dataset
- The dataset contains soccer players’ match statistics with the target column **`rating`**.
- Example features:
  - Goals, Assists, Minutes Played
  - Passing accuracy, Shot accuracy
  - Tackles, Interceptions, Fouls
  - Role (Attacker, Midfielder, Defender, Goalkeeper)

---

## 🔑 Key Steps in the Project

### 1. Data Preprocessing
- Dropped irrelevant columns (`name`, `player_id`)
- Removed rows with missing values in `rating` and `role`
- One-hot encoded `role`
- Converted percentage fields (`pass_success`, `shot_accuracy`) into numeric
- Filled remaining NaN values with `0`
- Scaled features using **MinMaxScaler**

### 2. Exploratory Data Analysis
- Correlation heatmaps of features vs rating  
- Boxplots for impact of `goals`, `chances_created` on `rating`  
- Visualization of missing values  

### 3. Model Building
- **Neural Network Architecture**
  - Dense (20 neurons, ReLU) + Dropout(0.25)
  - Dense (10 neurons, ReLU) + Dropout(0.25)
  - Dense (5 neurons, ReLU)
  - Dense (1 neuron, Linear) → Output rating
- Optimizer: **Adam (lr=0.001)**
- Loss function: **MSE**
- Early stopping to prevent overfitting

### 4. Model Evaluation
- **Mean Absolute Error (MAE)**
- 0.3950795469065728
- **Mean Squared Error (MSE)**
- -0.26269558633524465
  

### 5. Prediction on New Player
You can input custom player statistics and predict their **rating**:

```python
new_player = [
    0, 0, 0, 90, 0, 0, 0, 5, 50, 0, 2, 0,
    80, 5, 1, 50, 1, 2, 0, 15, 20, 8, 8, 70,
    75, 4, 8, 8, 4, 10, 8, 6, 8, 2, 7, 15,
    10, 8, 3, 0, 0, 1, 1, 0, 0, 0, 1
]

new_rating = predict_player_rating(new_player)
print(new_rating)
