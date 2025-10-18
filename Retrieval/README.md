-------------------------主要文件---------------------------

config.py 是配置文件

build.py 提取图像特征

main.py fastapi 部署服务

-------------------------使用---------------------------

0. 修改 config.py

- 根据注释修改！！

1. 首先 创建并启动环境（只需运行一次）

```
conda create -n text-image-search python=3.12
conda activate text-image-search
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

1. 第一步 提取图像特征（只需运行一次）

conda activate text-image-search && python build.py

3. 第二步 启动服务

conda activate text-image-search && python main.py



4. 测试

参考 test.py 中的python代码！！
