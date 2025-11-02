# 🔋 Battery State of Health (SOH) Prediction using LSTM

This project provides a **Streamlit web app** to visualize and predict the **State of Health (SOH)** of lithium-ion batteries using an **LSTM (Long Short-Term Memory)** neural network.  
The model is trained on cycle-based degradation features such as charge/discharge current, voltage, temperature, and cycle count.

---

## 🧠 Project Overview

- **Training Model:** LSTM-based deep learning model trained on cycle-wise data.
- **Input:** Features from the last 20 cycles:
  ```
  ['chI', 'chV', 'chT', 'disI', 'disV', 'disT', 'BCt']
  ```
- **Output:** Predicted **State of Health (SOH)** (in real scale).

The app allows users to upload a `.csv` or `.xlsx` file containing at least the **last 20 cycles** of these features for SOH prediction.

---

## 📁 Project Structure

```
battery_soh_lstm/
│
├── battery_validation_app.py     # Streamlit app for testing/validation
├── battery_lstm_model.h5         # Trained LSTM model (saved from training phase)
├── scaler.pkl                    # MinMaxScaler object used during training
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── sample_input.csv              # Example input file for testing
```

---

## ⚙️ Installation

### 1️⃣ Create and activate a virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate       # on Windows
# OR
source myenv/bin/activate      # on Mac/Linux
```

### 2️⃣ Install dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

If you get `[WinError 32]` or permission issues:
- Close all Explorer and Python windows.
- Delete `myenv\Lib\site-packages\streamlit` if partially installed.
- Re-run the install command above.

---

## 🧩 Required Dependencies

Add these to `requirements.txt`:

```
streamlit>=1.30.0
tensorflow>=2.10.0
scikit-learn>=1.2.0
pandas>=1.4.0
numpy>=1.23.0
matplotlib>=3.7.0
```

---

## 🚀 Running the App

Once dependencies are installed and your environment is active:

```bash
streamlit run battery_validation_app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 📊 Input File Format

Upload a `.csv` or `.xlsx` file containing **at least 20 recent cycles** with these columns:

| chI | chV | chT | disI | disV | disT | BCt |
|------|------|------|-------|-------|-------|------|
| 1.34 | 4.21 | 26.9 | 2.07 | 3.13 | 32.08 | 216 |
| 1.26 | 4.14 | 26.4 | 2.04 | 3.72 | 32.48 | 215 |
| ...  | ...  | ...  | ...  | ...  | ...  | ...  |

---

## 🧪 Output

The app displays:

- The **scaled and actual SOH prediction** for the given input data.
- Optional visualization of the input cycle data.

---

## 🧰 Troubleshooting

**Error:** `streamlit: command not found`  
👉 Ensure your virtual environment is activated and Streamlit installed:
```bash
myenv\Scripts\activate
pip install streamlit
```

**Error:** `[WinError 32] The process cannot access the file...`  
👉 Close all programs using the project folder (Explorer, Python IDEs) and retry installation.

---

## 📜 License

MIT License © 2025 — Developed by Gurunathagouda.

---

## 🙌 Acknowledgments

- Dataset: [Lithium-ion Battery Degradation Dataset (Kaggle)](https://www.kaggle.com/datasets/programmer3/lithium-ion-battery-degradation-dataset)
- Frameworks: TensorFlow, Streamlit, scikit-learn
