# 🎓 Student Performance Prediction System

A modern, interactive web application that predicts students' final examination grades using **Linear Regression**, built with **Python**, **Streamlit**, and **Scikit-learn**.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn&logoColor=white)

---

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the App](#running-the-app)
- [Screenshots](#screenshots)
- [Technologies](#technologies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 📖 About

This project uses the **UCI Student Performance (Mathematics)** dataset to train a Linear Regression model that predicts a student's final grade (**G3**, scale 0–20) based on five features:

| Feature | Description |
|---------|-------------|
| `studytime` | Weekly study time (1–4) |
| `failures` | Number of past class failures (0–4) |
| `absences` | Number of school absences |
| `G1` | First period grade (0–20) |
| `G2` | Second period grade (0–20) |

---

## ✨ Features

- **Dashboard** — KPI cards, interactive Plotly charts, prediction history
- **Predict Performance** — Input form → Linear Regression prediction → gauge chart
- **Student Analytics** — Grade distributions, study-time analysis, correlation heatmap
- **Dataset Explorer** — Preview, filter, search, download data
- **Model Information** — ML concepts, evaluation metrics, residual plots, coefficients
- **About Project** — Abstract, objectives, technologies, dataset source
- **Prediction History** — Session-based log with CSV export
- **Premium Dark UI** — Custom CSS theme with navy/blue/purple palette

---

## 📁 Project Structure

```
Student_Performance_Prediction/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   └── student-mat.csv     # UCI dataset (you must download this)
│
├── model/
│   └── train_model.py      # ML pipeline (load, train, predict)
│
└── assets/
    └── style.css           # Custom CSS theme
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.9 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** (included with Python)

### Steps

1. **Clone or download** this project folder.

2. **Open a terminal** and navigate to the project directory:
   ```bash
   cd Student_Performance_Prediction
   ```

3. **(Recommended)** Create a virtual environment:
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Dataset Setup

The dataset is **not** included in this repository. You must download it manually:

1. Visit the UCI Machine Learning Repository:
   **https://archive.ics.uci.edu/dataset/320/student+performance**

2. Click **Download** to get the ZIP file (`student.zip`).

3. Extract the archive and locate **`student-mat.csv`**.

4. Copy `student-mat.csv` into the `data/` folder of this project:
   ```
   Student_Performance_Prediction/
   └── data/
       └── student-mat.csv   ← place it here
   ```

> **Note:** The file uses a **semicolon (`;`)** as separator. The code handles this automatically.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The application will open in your default browser at **http://localhost:8501**.

---

## 🛠️ Technologies

| Technology | Purpose |
|-----------|---------|
| Python 3.9+ | Programming language |
| Streamlit | Web application framework |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine Learning (Linear Regression) |
| Plotly | Interactive visualizations |
| Matplotlib | Static plots (optional) |
| HTML / CSS | Custom UI theming |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: student-mat.csv` | Download the dataset and place it in `data/` (see [Dataset Setup](#dataset-setup)). |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again. |
| Streamlit not found | Make sure your virtual environment is activated. |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502`. |
| Charts not rendering | Upgrade Plotly: `pip install --upgrade plotly`. |
| Blank page on mobile | Use a modern browser (Chrome, Firefox, Edge). |

---

## 📄 License

This project is for **educational purposes**. The dataset is provided by the UCI Machine Learning Repository under its own terms.

---

> Built with ❤️ for Data Science education.
