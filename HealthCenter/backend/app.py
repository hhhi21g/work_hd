from flask import Flask, jsonify, render_template, send_from_directory, request
import pymysql
import os
import sys

import get_pic

app = Flask(__name__)

# 设置保存图片的静态目录
GENERATED_DIR = os.path.join("static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


# @app.route("/generate_image", methods=["POST"])
# def generate():
#     data = request.json
#     prompt = data.get("prompt")
#     if not prompt:
#         return jsonify({"error": "需要提供 prompt"}), 400
#
#     filename = f"{hash(prompt)}.png"
#     output_path = os.path.join(GENERATED_DIR, filename)
#
#     get_pic.generate_image(prompt, output_path)
#     return jsonify({"url": f"/static/generated/{filename}"})


def get_db_connection():
    return pymysql.connect(
        host='212.129.223.4',
        user='root',
        password='nineone4536251',
        database='health',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


@app.route('/api/news')
def api_news():
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute(
        "SELECT title, content, publish_time, url, image_filename FROM news ORDER BY publish_time DESC LIMIT 10")
    result = cursor.fetchall()
    cursor.close()
    db.close()

    for item in result:
        if item['content']:
            item['summary'] = item['content'][:100] + '...'
        else:
            item['summary'] = "（暂无正文内容）"

        # 获取新闻图片的静态路径，如果没有图片，则使用默认图片
        image_filename = item.get('image_filename')
        if image_filename and os.path.exists(os.path.join("static", "images", image_filename)):
            item['image_url'] = f"/static/images/{image_filename}"
        else:
            item['image_url'] = "/static/default.jpg"  # 默认图片

    return jsonify(result)


# API 路由：返回最新10条新闻
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
