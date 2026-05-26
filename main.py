#Better to read all comments for understanding. Thankyou in advance and don't forget to support my channel!
from ml.MainInterface import MlAlgorithmInterface

from data.MainInterface import DataInterface
from data.UtilsForDataGather import load_csv
from data.UtilsForDataPrepare import rows_to_XY, parse_datetime_features, denormalize_y, normalize_point, train_test_split

from evaluation.Metrics import evaluate
from evaluation.Visualization import plot_predictions, plot_errors

from datetime import datetime, timezone, timedelta
import pandas as pd

#Initialize ml algorithms interface
MODEL_TYPE = "xgboost" #Change model type to choose different type ["linear", "logistic", "random_forest", "xgboost"]
ml_interface = MlAlgorithmInterface()

#Initialize data interface |CHANGE ACCORDING TO DATA SOURCE|
DATA_SOURCE = "AlbionDataProject" #Change name of data source to initialize new data source just use command "python data/NewDataSource.py <name>"
DATA_SOURCE_FUNCTION_USE = "fetch_history" # Change function name according to data source function which are able
data_interface = DataInterface()

#Shows all available sources and functions
data_interface.info()
print()

#Fetch and save data in directory data/collected/<name>/ in .csv file
rows2, data_path = data_interface.use_source_function(DATA_SOURCE, DATA_SOURCE_FUNCTION_USE, save=True) # |CHANGE ACCORDING TO DATA SOURCE|
data = load_csv(data_path)

#Fetch data without save
rows = data_interface.use_source_function(DATA_SOURCE, DATA_SOURCE_FUNCTION_USE, save=False) # |CHANGE ACCORDING TO DATA SOURCE|

#Filter and convert data in dataframe |CHANGE ACCORDING TO DATA SOURCE|
df = pd.DataFrame(rows)
df = df[df['location'].isin(["Thetford"])] #Accordint on date you can change specific city or other stuff! With dataframe it is easier
data = df.to_dict(orient='records')

#Prepare data for X parameters and Y points |CHANGE ACCORDING TO DATA SOURCE|
X, y, norm_params = rows_to_XY(
    rows=data, #All collected data
    feature_cols=["quality"], #Numeric columns
    date_cols=["timestamp"], #Date columns (separeta because need to normalize)
    label_col="avg_price", #Numeric colum which will goes to Y data for future predicts
    normalize=True, #To convert data in to float from 0.0 to 1.0 for models trained if false will return raw splited data
    scaler="mean" #Convert type there is 2 options "minmax" or "mean"
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Train model
model = ml_interface.train_model(X_train, y_train, MODEL_TYPE)
#Save model in directory models
model_name = ml_interface.save_model(model, norm_params, MODEL_TYPE)
# Load saved model
model, norm_params = ml_interface.load_model(model_name)

#Model Test Results
train_preds = ml_interface.predict_model(model, X_test, MODEL_TYPE)
y_pred = [r["value"] for r in train_preds]
evaluate(MODEL_TYPE, y_test, y_pred, norm_params, denormalize_y)

#Model prediction and errors visualization
plot_predictions(y_test, y_pred, norm_params, denormalize_y)
plot_errors(y_test, y_pred, norm_params, denormalize_y)

#Prepare new endpoints on today with GMT+3 time ~can change hours with higher or plus hours |CHANGE ACCORDING TO DATA SOURCE|
time_now = datetime.now(timezone(timedelta(hours=11)))
normalize_date = parse_datetime_features(str(time_now), normalize=True)
new_points_raw = [[float(i)] + normalize_date for i in range(1,3)]
new_points = [normalize_point(p, norm_params) for p in new_points_raw]

#Using model for predict and then denormalize data to recive proper result! Raw result will show value in range from 0.0 to 1.0
results = ml_interface.predict_model(model, new_points, MODEL_TYPE)
print(f"Model Predictions on {time_now.strftime('%H:%M:%S %m-%d-%Y')}:")
for i, res in enumerate(results):
    print(f"quality {i+1} {denormalize_y(res['value'], norm_params):.0f} silver")