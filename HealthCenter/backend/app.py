from flask import Flask, jsonify, render_template, send_from_directory, request
import pymysql
import os
import sys

import get_pic

app = Flask(__name__)

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
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT title, content, publish_time, url FROM knowledges ORDER BY publish_time DESC LIMIT 10")
    result = cursor.fetchall()
    cursor.close()
    db.close()

    # 提取正文摘要（前60字）
    for item in result:
        if item['content']:
            item['summary'] = item['content'][:100] + '...'
        else:
            item['summary'] = "（暂无正文内容）"

    return jsonify(result)


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


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


if __name__ == '__main__':
    app.run(debug=True)


