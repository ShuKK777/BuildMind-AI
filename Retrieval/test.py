import requests

# API 地址
url = "http://localhost:8848/semantic_search"

# 请求数据
payload = {
    "text_query": "office,high,Modern Office Building,single,trees road,eye_level",
    "top_k": 2,
    # "filters": {
    #     "height_level_int": 10,  # 可以添加多个
    # }
}

# 发送 POST 请求
response = requests.post(url, json=payload)

# 打印结果
print("状态码:", response.status_code)
print("返回结果:", response.json())


# 访问图片 http://localhost:8848/images/mob_001.jpeg