import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory, request
import pymysql
import os
import sys
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import joblib
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

UPLOAD_FOLDER = 'static\\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置保存图片的静态目录
GENERATED_DIR = os.path.join("static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


def get_db_connection():
    return pymysql.connect(
        host='212.129.223.4',
        user='root',
        password='nineone4536251',
        database='health',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


@app.route('/api/home/news')
def api_home_news():
    selected_ids = [1508, 1529, 1591, 1618, 1926, 1939, 1959, 2634, 2070, 2004]

    db = get_db_connection()
    cursor = db.cursor()

    # 使用 IN 子句查询指定 ID 的新闻数据
    cursor.execute(
        f"SELECT title, content, publish_time, url FROM news WHERE id IN ({','.join(map(str, selected_ids))}) ORDER BY FIELD(id, {','.join(map(str, selected_ids))})")
    result = cursor.fetchall()
    cursor.close()
    db.close()

    for index, item in enumerate(result):
        if item['content']:
            item['summary'] = item['content'][:100] + '...'
        else:
            item['summary'] = "（暂无正文内容）"

        # 根据新闻的顺序为每条新闻分配静态图片路径
        image_filename = f"a{index + 1}.jpg"  # 从 a1.jpg 开始
        image_path = os.path.join("static", "carousel", image_filename)

        if os.path.exists(image_path):
            item['image_url'] = f"/static/carousel/{image_filename}"
        else:
            item['image_url'] = "/static/default.jpg"  # 默认图片

    return jsonify(result)


@app.route('/api/home2/news')
def api_home2_news():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT title, url FROM news ORDER BY publish_time DESC LIMIT 15")
    result = cursor.fetchall()
    cursor.close()
    db.close()
    return jsonify(result)


@app.route('/api/home/policies')
def api_home_policies():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT title, url FROM policy ORDER BY publish_time DESC LIMIT 15")
    result = cursor.fetchall()
    cursor.close()
    db.close()
    return jsonify(result)


@app.route('/api/home/knowledges')
def api_home_knowledges():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT title, url FROM knowledges ORDER BY publish_time DESC LIMIT 15")
    result = cursor.fetchall()
    cursor.close()
    db.close()
    return jsonify(result)


@app.route('/api/home/notices')
def api_home_notices():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT title, url FROM notice ORDER BY publish_time DESC LIMIT 15")
    result = cursor.fetchall()
    cursor.close()
    db.close()
    return jsonify(result)


@app.route('/api/knowledges')
def api_knowledges():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))
    offset = (page - 1) * size

    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM knowledges")
    total = cursor.fetchone()['total']

    # 获取当前页数据
    cursor.execute(
        "SELECT title, content, publish_time, url FROM knowledges ORDER BY publish_time DESC LIMIT %s OFFSET %s",
        (size, offset))

    result = cursor.fetchall()
    cursor.close()
    db.close()

    # 提取正文摘要（前60字）
    for item in result:
        if item['content']:
            item['summary'] = item['content'][:100] + '...'
        else:
            item['summary'] = "（暂无正文内容）"

    return jsonify({
        "total": total,
        "data": result
    })


@app.route('/api/news')
def api_news():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))
    offset = (page - 1) * size

    db = get_db_connection()
    cursor = db.cursor()

    # 获取总数量
    cursor.execute("SELECT COUNT(*) AS total FROM news")
    total = cursor.fetchone()['total']

    # 获取当前页数据
    cursor.execute("SELECT title, content, publish_time, url FROM news ORDER BY publish_time DESC LIMIT %s OFFSET %s",
                   (size, offset))
    result = cursor.fetchall()
    cursor.close()
    db.close()

    for item in result:
        item['summary'] = (item['content'][:100] + '...') if item['content'] else "（暂无正文内容）"

    return jsonify({
        "total": total,
        "data": result
    })


@app.route('/api/notice')
def api_notice():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))
    offset = (page - 1) * size

    db = get_db_connection()
    cursor = db.cursor()

    # 获取总数量
    cursor.execute("SELECT COUNT(*) AS total FROM notice")
    total = cursor.fetchone()['total']

    # 获取当前页数据
    cursor.execute("SELECT title, content, publish_time, url FROM notice ORDER BY publish_time DESC LIMIT %s OFFSET %s",
                   (size, offset))
    result = cursor.fetchall()
    cursor.close()
    db.close()

    for item in result:
        item['summary'] = (item['content'][:100] + '...') if item['content'] else "（暂无正文内容）"

    return jsonify({
        "total": total,
        "data": result
    })


@app.route('/api/policy')
def api_policy():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))
    offset = (page - 1) * size

    db = get_db_connection()
    cursor = db.cursor()

    # 获取总数量
    cursor.execute("SELECT COUNT(*) AS total FROM policy")
    total = cursor.fetchone()['total']

    # 获取当前页数据
    cursor.execute("SELECT title, content, publish_time, url FROM policy ORDER BY publish_time DESC LIMIT %s OFFSET %s",
                   (size, offset))
    result = cursor.fetchall()
    cursor.close()
    db.close()

    for item in result:
        item['summary'] = (item['content'][:100] + '...') if item['content'] else "（暂无正文内容）"

    return jsonify({
        "total": total,
        "data": result
    })


@app.route('/')
def home():
    return render_template('home.html', active_page='home')


@app.route('/news')
def news():
    return render_template('news.html', active_page='news')


@app.route('/notice')
def notice():
    return render_template('notice.html', active_page='notice')


@app.route('/policy')
def policy():
    return render_template('policy.html', active_page='policy')


@app.route('/knowledge')
def knowledge():
    return render_template('knowledge.html', active_page='knowledge')


@app.route('/application')
def application():
    return render_template('application.html', active_page='application')


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


@app.route('/bmi')
def bmi():
    return render_template('bmi.html', active_page='bmi')


@app.route('/water-intake')
def water_intake():
    return render_template('water-intake.html', active_page='water-intake')


@app.route('/sleep-quality')
def sleep_quality():
    return render_template('sleep-quality.html', active_page='sleep-quality')


@app.route('/emergency-guide')
def emergency_guide():
    return render_template('emergency-guide.html', active_page='emergency-guide')


@app.route('/info-guide')
def info_guide():
    return render_template('info-guide.html', active_page='info-guide')


@app.route('/chest-diagnosis', methods=['GET', 'POST'])
def chest_diagnosis():
    result = None

    # 14 类标签（请根据你的模型调整）
    class_names = [
        '肺不张／肺萎陷',
        '心脏肥大',
        '胸腔积液',
        '渗透',
        '肿块',
        '结节',
        '肺炎',
        '气胸',
        '实变',
        '浮肿／水肿',
        '气肿／肺气肿',
        '纤维化',
        '胸膜增厚',
        '疝气'
    ]

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 修改成模型需要的尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # 加载模型（仅第一次加载）
    global model_chest
    if 'model_chest' not in globals():
        model_chest = models.resnet18(num_classes=14)
        model_chest.load_state_dict(torch.load('models/chest_model.pth', map_location='cpu'))
        model_chest.eval()

    if request.method == 'POST':
        file = request.files.get('ct_image')
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # 图像预处理
            image = Image.open(save_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

            # 模型推理
            with torch.no_grad():
                output = model_chest(input_tensor)
                prob = torch.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(prob).item()
                pred_label = class_names[pred_idx]
                confidence = f"{prob[pred_idx].item() * 100:.2f}"

            result = {
                'label': pred_label,
                'confidence': confidence,
                'filename': filename
            }

    return render_template('chest-diagnosis.html', active_page='chest-diagnosis', result=result)


@app.route('/diabetes-diagnosis', methods=['GET', 'POST'])
def diabetes_diagnosis():
    result = None

    # 定义模型结构（必须与训练时一致）
    # 正确的模型结构（与你训练时完全一致）
    class DiabetesNet(nn.Module):
        def __init__(self):
            super(DiabetesNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, 16),
                nn.ReLU(),

                nn.Linear(16, 1),

                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    # 加载模型（仅第一次）
    global diabetes_model
    if 'diabetes_model' not in globals():
        diabetes_model = DiabetesNet()
        diabetes_model.load_state_dict(torch.load("models/diabetes_model.pth", map_location='cpu'))
        diabetes_model.eval()

    # 加载标准化器（仅第一次）
    global diabetes_scaler
    if 'diabetes_scaler' not in globals():
        import joblib
        diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")

    if request.method == 'POST':
        try:
            # 读取表单数据并转为 float
            user_input = [
                float(request.form.get("Pregnancies")),
                float(request.form.get("Glucose")),
                float(request.form.get("BloodPressure")),
                float(request.form.get("SkinThickness")),
                float(request.form.get("Insulin")),
                float(request.form.get("BMI")),
                float(request.form.get("DiabetesPedigreeFunction")),
                float(request.form.get("Age"))
            ]

            # 标准化
            input_scaled = diabetes_scaler.transform([user_input])  # shape: [1, 8]
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            # 模型预测
            with torch.no_grad():
                output = diabetes_model(input_tensor)
                prob = output.item()
                label = "患糖尿病" if prob >= 0.5 else "未患糖尿病"

            result = {
                "label": label,
                "confidence": f"{prob * 100:.2f}"
            }

        except Exception as e:
            result = {
                "label": "输入错误",
                "confidence": "0.00"
            }

    return render_template("diabetes-diagnosis.html", active_page="diabetes-diagnosis", result=result)


@app.route('/heart-diagnosis', methods=['GET', 'POST'])
def heart_diagnosis():
    result = None

    # 模型加载（只加载一次）
    global heart_pipeline
    if 'heart_pipeline' not in globals():
        heart_pipeline = joblib.load('models/heart_disease_model.pkl')  # 路径按你实际情况修改

    if request.method == 'POST':
        try:
            # 1. 收集用户输入
            user_data = {
                "Age": float(request.form.get("Age")),
                "Sex": request.form.get("Sex"),
                "Chest pain type": request.form.get("Chest pain type"),
                "BP": float(request.form.get("BP")),
                "Cholesterol": float(request.form.get("Cholesterol")),
                "FBS over 120": int(request.form.get("FBS over 120")),
                "EKG results": request.form.get("EKG results"),
                "Max HR": float(request.form.get("Max HR")),
                "Exercise angina": request.form.get("Exercise angina"),
                "ST depression": float(request.form.get("ST depression")),
                "Slope of ST": request.form.get("Slope of ST"),
                "Number of vessels fluro": float(request.form.get("Number of vessels fluro")),
                "Thallium": request.form.get("Thallium")
            }

            # 2. 转为 DataFrame（模型接受 DataFrame 输入）
            df_input = pd.DataFrame([user_data])

            # 3. 推理
            prob = heart_pipeline.predict_proba(df_input)[0][1]
            label = "有心脏病风险" if prob >= 0.5 else "风险较低"

            result = {
                "label": label,
                "confidence": f"{prob * 100:.2f}"
            }

        except Exception as e:
            result = {
                "label": "输入错误或模型故障",
                "confidence": "0.00"
            }

    return render_template("heart-diagnosis.html", active_page="heart-diagnosis", result=result)


if __name__ == '__main__':
    app.run(debug=True)
