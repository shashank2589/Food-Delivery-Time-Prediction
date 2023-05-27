from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

application=Flask(__name__)
app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Vehicle_condition = int(request.form.get('Vehicle_condition')),
            multiple_deliveries = float(request.form.get('multiple_deliveries')),
            Festival = float(request.form.get('Festival')),
            distance = float(request.form.get('distance')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density= request.form.get('Road_traffic_density'),
            City = request.form.get('City')
        )
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=round(pred[0],2)

        return render_template('result.html',final_result=result)
    

@app.route('/train')
def train_pipeline():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initaite_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    best_model_name, best_model_score, train_model_report, test_model_report = model_trainer.initate_model_training(train_arr, test_arr)

    return render_template('train_pipeline.html', best_model_name=best_model_name, best_model_score=best_model_score, train_model_report=train_model_report, test_model_report=test_model_report)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=8080)