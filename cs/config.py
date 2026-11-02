"""客服 Agent 配置 —— 复用根目录 .env 的密钥，客服场景调低温度保证稳定"""
from config import Config


class CSConfig:
    API_KEY = Config.DEEPSEEK_KEY
    BASE_URL = Config.BASE_URL
    MODEL = Config.MODEL

    # 客服场景：温度必须低。政策类回答需要确定性，0.1 可显著降低措辞漂移与编造倾向
    TEMPERATURE = 0.1

    # 单会话历史保留轮数（user+assistant 记 2 条）
    HISTORY_MAX = 10
