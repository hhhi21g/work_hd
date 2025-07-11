import pymysql

try:
    db = pymysql.connect(
        host='212.129.223.4',
        user='root',
        password='nineone4536251',
        database='health'
    )
    print("连接成功！")
    db.close()
except Exception as e:
    print("连接失败：", e)
