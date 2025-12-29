# Advanced-Time-Series-Forecasting-with-Neural-ODEs-Neural-Ordinary-Differential-Equations-
This project focuses on forecasting time series data using Neural ODEs by learning patterns in continuous time instead of fixed time steps. It compares LSTM with Neural ODE, Neural CDE, and Latent ODE models to show how continuous-time approaches handle complex and real-world data more effectively.

# Neural ODE for Time Series Forecasting

This repository presents a complete comparative study between a **discrete-time LSTM model** and **continuous-time Neural Ordinary Differential Equation (Neural ODE)** models for non-linear time series forecasting. The work is extended to include **Neural Controlled Differential Equations (Neural CDEs)** and **Latent ODEs**, providing a modern continuous-time deep learning perspective.

---

## 📌 Project Objectives

- Prepare and preprocess a non-linear real-world time series dataset
- Implement an LSTM as a benchmark forecasting model
- Implement a Neural ODE for continuous-time modeling
- Extend the analysis to Neural CDEs and Latent ODEs
- Compare models using quantitative metrics and qualitative discussion
- Highlight challenges in learning continuous-time dynamics

---

## 📊 Dataset

- **Dataset:** AirPassengers (monthly airline passenger counts)
- **Source:** `statsmodels.datasets`
- **Characteristics:**
  - Non-linear trend
  - Seasonal patterns
  - Non-stationary behavior

---

## 🧠 Models Implemented

### 1. LSTM (Benchmark)
- Discrete-time recurrent neural network
- Fixed time-step transitions
- Strong baseline for time series forecasting

### 2. Neural ODE
- Continuous-time hidden state evolution
- Uses numerical ODE solvers (`torchdiffeq`)
- Learns system dynamics instead of discrete transitions

### 3. Neural CDE (Extension)
- Models data as a continuous control signal
- Handles irregularly sampled time series naturally
- Improves expressiveness over Neural ODEs

### 4. Latent ODE (Extension)
- Combines Neural ODEs with Variational Autoencoders (VAE)
- Learns latent continuous trajectories
- Suitable for noisy or partially observed time series

---

## ⚙️ Tech Stack

- Python
- PyTorch
- torchdiffeq
- NumPy, Pandas
- scikit-learn
- Matplotlib
- ReportLab (for PDF generation)

---

## 📈 Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## 📂 Repository Structure

