from flask import Flask, render_template, jsonify

app = Flask(__name__)  # 创建Flask应用


@app.route('/')  # 定义路由: 根路径
def index():
    return render_template('index.html', title='一周走势页面')


@app.route('/api/data')
def data():
    days = [1, 2, 3, 4, 5, 6, 7]
    sales = [120, 200, 150, 80, 70, 110, 130]
    return jsonify({'days': days, 'sales': sales})


if __name__ == '__main__':
    # 允许其他主机使用本机IP进行访问,同一局域网
    app.run(debug=True, host='0.0.0.0', port=5000)  # 开启开发模式,默认在http://127.0.0.1:5000
