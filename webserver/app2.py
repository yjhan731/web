import flask
from flask import Flask, request, jsonify
from gevent import monkey, pywsgi
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import numpy as np
import joblib
import pandas as pd
import os
import csv  # 导入 CSV 库

# 加载模型和标量器
fzy_scaler = joblib.load('./fzy_scaler.pkl')
fzy_model = joblib.load('./fzy_model.pkl')

app = Flask(__name__)
CORS(app)

# 定义数据存储的 CSV 文件路径
csv_file_path = './collect_data.csv'

# 检查 CSV 文件是否存在，如果不存在则创建并添加表头
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Time', 'Id', 'Sex', 'Age', 'BMI', 'DBP', 'SBP', 'HeartRate', 'HbA1c', 'FBG',
                      'TG', 'AST', 'WBC', 'HGB', 'PLT', 'CR', 'RBC', 'LDL', 'ALT', 'BUN', 'HDL-C', 'TC', 'AEYW', 'AEYW_pred']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json  # 获取POST请求中的JSON数据

    # 处理数据
    cate = {'Sex': float(data['sex']), 'BUN': float(data['bun']), 'HDL': float(data['hdl']), 'TC': float(data['tc'])}
    cont = {'Age': float(data['age']), 'BMI': float(data['bmi']), 'HbA1c': float(data['hba1c']), 'FBG': float(data['fbg']),
            'SBP': float(data['sbp']), 'DBP': float(data['dbp']), 'Heart rate': float(data['heartRate'])}
    std_fea = np.array(
        [cont['Age'], cont['BMI'], cont['HbA1c'], cont['FBG'], cont['SBP'], cont['DBP'], cont['Heart rate']])
    std_fea_reshape = std_fea.reshape(1, -1)
    cont_scaled = fzy_scaler.transform(std_fea_reshape)
    input_data = np.array(
        [cate['Sex'], cate['BUN'], cont_scaled[0][6], cont_scaled[0][3], cate['HDL'], cont_scaled[0][2],
         cont_scaled[0][1], cont_scaled[0][0], cate['TC'], cont_scaled[0][4], cont_scaled[0][5]])
    input_data = input_data.reshape(1, -1)
    y_pred_proba1 = fzy_model.predict_proba(input_data)[:, 1]
    result = round(y_pred_proba1[0], 4) * 100

    # 获取记录时间
    from datetime import date
    today = date.today()
    t = today.strftime("%Y-%m-%d")

    # 准备新数据
    new_data = {
        'Name': data['name'],
        'Time': t,
        'Id': data['id'],
        'Sex': data['sex'],
        'Age': data['age'],
        'BMI': data['bmi'],
        'DBP': data['dbp'],
        'SBP': data['sbp'],
        'HeartRate': data['heartRate'],
        'HbA1c': data['hba1c'],
        'FBG': data['fbg'],
        'TG': data['tg'],
        'AST': data['ast'],
        'WBC': data['wbc'],
        'HGB': data['hgb'],
        'PLT': data['plt'],
        'CR': data['cr'],
        'RBC': data['rbc'],
        'LDL': data['ldl'],
        'ALT': data['alt'],
        'BUN': data['bunf'],
        'HDL-C': data['hdlf'],
        'TC': data['tcf'],
        'AEYW': data['aeyw'],
        'AEYW_pred': round(result, 2)
    }

    # 读取CSV文件为DataFrame，将新数据追加到DataFrame，并保存回CSV文件
    df = pd.read_csv(csv_file_path)
    #df = df.append(new_data, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(csv_file_path, index=False)

    # 返回响应
    response = {'message': round(result, 2)}
    print("Received data:", data)
    return jsonify(response)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)  # 在调试模式下运行，方便调试
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
