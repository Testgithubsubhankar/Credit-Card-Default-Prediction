from flask import Flask, request
import sys

import pip
from creditcard.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from creditcard.logger import logging
from creditcard.exception import CreditcardException
import os, sys
import json
from creditcard.config.configuration import Configuration
from creditcard.constant import CONFIG_DIR, get_current_time_stamp
from creditcard.pipeline.pipeline import Pipeline
from creditcard.entity.housing_predictor import CreditcardPredictor,CreditcardData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "creditcard"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from creditcard.logger import get_log_dataframe

CREDITCARD_DATA_KEY = "creditcard_data"
CREDITCARD_DEFAULT_PAYMENT_NEXT_MONTH_KEY = "creditcard_default_payment_next_month"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'housing'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("housing", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        CREDITCARD_DATA_KEY : None,
        CREDITCARD_DEFAULT_PAYMENT_NEXT_MONTH_KEY : None
    }

    if request.method == 'POST':
        id = float(request.form['id'])
        limit_bal = float(request.form['limit_bal'])
        age = float(request.form['age'])
        bill_amt1 = float(request.form['bill_amt1'])
        bill_amt2 = float(request.form['bill_amt2'])
        bill_amt3 = float(request.form['bill_amt3'])
        bill_amt4 = float(request.form['bill_amt4'])
        bill_amt5 = float(request.form['bill_amt5'])
        bill_amt6 = float(request.form['bill_amt6'])
        pay_amt1 = float(request.form['pay_amt1'])
        pay_amt2 = float(request.form['pay_amt2'])
        pay_amt3 = float(request.form['pay_amt3'])
        pay_amt4 = float(request.form['pay_amt4'])
        pay_amt5 = float(request.form['pay_amt5'])
        pay_amt6 = float(request.form['pay_amt6'])
        sex = request.form['sex']
        education = request.form['education']
        marriage = request.form['marriage']
        pay_0 = request.form['pay_0']
        pay_2 = request.form['pay_2']
        pay_3= request.form['pay_3']
        pay_4 = request.form['pay_4']
        pay_5 = request.form['pay_5']
        pay_6 = request.form['pay_6']
        default_payment_next_month = request.form['default_payment_next_month']

        creditcard_data = CreditcardData(id=id,
                                   limit_bal=limit_bal,
                                   age= age,
                                   bill_amt1= bill_amt1,
                                   bill_amt2= bill_amt2,
                                   bill_amt3=bill_amt3,
                                   bill_amt4= bill_amt4,
                                   bill_amt5= bill_amt5,
                                   bill_amt6=bill_amt6,
                                   pay_amt1=pay_amt1,
                                   pay_amt2=pay_amt2,
                                   pay_amt3=pay_amt3,
                                   pay_amt4=pay_amt4,
                                   pay_amt5=pay_amt5,
                                   pay_amt6=pay_amt6,
                                   sex=sex,
                                   education=education,
                                   marriage=marriage,
                                   pay_0= pay_0,
                                   pay_2=pay_2,
                                   pay_3=pay_3,
                                   pay_4=pay_4,
                                   pay_5= pay_5,
                                   pay_6 = pay_6,
                                   default_payment_next_month= default_payment_next_month,
                                   )
        creditcard_df = creditcard_data.get_creditcard_input_data_frame()
        creditcard_predictor = CreditcardPredictor(model_dir=MODEL_DIR)
        creditcard_default_payment_next_month = creditcard_predictor.predict(X=creditcard_df)
        context = {
            CREDITCARD_DATA_KEY: creditcard_data.get_creditcard_data_as_dict(),
            CREDITCARD_DEFAULT_PAYMENT_NEXT_MONTH_KEY: creditcard_default_payment_next_month,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()

