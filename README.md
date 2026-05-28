# MLFramework v0.01

A ML framework designed for **easy addition of new data sources** and **simple integration of new machine learning models**.

Currently implemented models: 
**Linear Regression | Logistic Regression | Random Forest | XGBoost**

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Decursusss/MLFramework.git
cd MLFramework
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```

`main.py` serves as a full example pipeline:

```
Data Collection → Data Preparation → Model Training → Visualization → Saved Model → Predictions
```

You can use it as a reference for building your own pipelines.

---

## ➕ Adding a New Data Source

Run the command:

```bash
python data/NewDataSource.py <name>
```

Replace `<name>` with your new source name. The script will automatically:
- Create a new folder under `data/sources/<name>/`
- Generate a starter Python file and `config.json`

After that, you only need to:
1. Adapt the `config.json` to your source
2. Implement the data fetching logic in the generated `.py` file

The models will immediately be able to work with the new data - no other changes needed.

---

## ➕ Adding a New ML Model

1. Create a new file in `ml/MLAlgorithms/`, e.g. `MyModel.py`
2. Implement your model class/logic inside it
3. Register it in `ml/MainInterface.py` - add a call to your model by name

That's it. The framework is designed to make this as easier as possible.

---

## 📁 Project Structure

```
MLAlgorithms/
│
├── data/
│   ├── collected/
│   │   └── AlbionData/
│   │       ├── ['T4_BAG']_history_20260526.csv
│   │       ├── ['T4_BAG']_history_20260527.csv
│   │       └── ['T4_BAG']_prices_20260526.csv
│   │
│   └── sources/
│       └── AlbionDataProject/
│           ├── AlbionDataProject.py
│           └── config.json
│       ├── MainInterface.py
│       ├── NewDataSource.py
│       ├── UtilsForDataGather.py
│       └── UtilsForDataPrepare.py
│
├── evaluation/
│   ├── reports/
│   ├── Metrics.py
│   └── Visualization.py
│
├── ml/
│   ├── MLAlgorithms/
│   │   ├── LinealRegression.py
│   │   ├── LogicalRegression.py
│   │   ├── RandomForest.py
│   │   └── XGBoost.py
│   └── MainInterface.py
│
├── models/
│   ├── linear.json
│   ├── logical.json
│   ├── random_forest.json
│   └── xgboost.json
│
├── .gitignore
├── main.py
└── requirements.txt
```

---

## 📦 Folder Descriptions

### `data/`
Contains everything related to data collection and preparation.

- **`collected/`** - raw collected data stored as CSV files, organized by source name (e.g. `AlbionData/`)
- **`sources/`** - data source modules!Each source has its own folder with a Python script and a `config.json`
  - `MainInterface.py` - entry point for data loading across all sources
  - `NewDataSource.py` - CLI tool for creating a new data source
  - `UtilsForDataGather.py` - helper utilities for fetching/gathering data
  - `UtilsForDataPrepare.py` - helper utilities for cleaning and transforming data

### `evaluation/`
Handles model evaluation and result visualization.

- **`reports/`** - generated image reports
- `Metrics.py` - metric calculation logic
- `Visualization.py` - plotting and visual output of results in reports

### `ml/`
Contains all machine learning logic.

- **`MLAlgorithms/`** - one file per algorithm! Adding a new model = dropping a new `.py` file here
- `MainInterface.py` - routes model calls by name!Register new models here after adding them

### `models/`
Stores trained model configurations/weights as JSON files, one per algorithm.

---

## 📊 Results

<img width="1199" height="400" alt="image" src="https://github.com/user-attachments/assets/c354b476-75cf-409a-8e34-61bf7b4a638d" />
<img width="307" height="101" alt="image" src="https://github.com/user-attachments/assets/a3cbceb4-45b1-41ac-8729-ec4b2a035972" />


---

## 🎥 More Information

For more details and video walkthroughs, visit the YouTube channel:

👉 [https://www.youtube.com/@bastrikins1](https://www.youtube.com/@bastrikins1)
