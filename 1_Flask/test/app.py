from flask import Flask, render_template

app = Flask(__name__)  # 创建Flask应用


@app.route('/')  # 定义路由: 根路径
def index():
    return render_template('index.html', title='一周走势页面')


if __name__ == '__main__':
    app.run(debug=True)  # 开启开发模式,默认在http://127.0.0.1:5000
