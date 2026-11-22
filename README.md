# 稀疏观测三维高斯写回工具包

本仓库提供一个可重复运行的证据控制写回工具包，用于在稀疏观测条件下对三维高斯表示的候选增量执行来源检查、区间证书、参数可辨识性、保护区扰动、候选暂存和动作级回滚。

## 内容

- src/minegs/：写回门控、来源检查、区间证书、可辨识子空间和压力案例生成代码。
- configs/case_matrix.csv：12类可重复的压力条件定义。
- tests/：模块级和最小案例测试。
- examples/：最小运行示例和输出格式说明。
- requirements.txt、pyproject.toml：Python依赖和安装配置。
- data/README.md：输入数据接口与隐私处理要求。

## 安装与测试

python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
pytest -q

Linux/macOS环境请使用对应的虚拟环境激活命令。运行环境需要Python 3.10或更高版本。

## 最小运行

python examples/run_minimal_case.py

程序输出动作状态、提交标志、证书状态、参数可辨识性诊断以及回滚审计字段。示例只使用内置合成输入，不包含原始采集数据。

## 数据接口

原始图像、点云、相机轨迹、三维高斯模型和运行产物不随代码包提供。使用者应在获得数据授权后，按照 data/README.md 中的字段约定建立本地数据目录，并避免将个人信息、受限数据和大体积模型提交到版本库。

