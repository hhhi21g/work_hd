from flask import Flask, render_template, jsonify, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.get_json()  # 解析前端发送的JSON数据为Python对象,典型用于POST
    name = data.get('name', '')
    if not name:
        return jsonify({'message': '输入不能为空'})
    return jsonify({'message': f'{name}: 成功接收'})


if __name__ == '__main__':
    app.run(debug=True)
