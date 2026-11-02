"""
配置层 —— 从 .env 文件 + 环境变量加载
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

# 加载系统环境变量中的 .env（如果用户在外面设了）
load_dotenv(override=False)


class Config:
    # ── AI 模型 ──
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    BASE_URL     = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    MODEL        = os.getenv("MODEL", "deepseek-v4-pro")
    TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.3"))

    # ── 系统提示词 ──
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or (
        "你是一个只能通过调用工具来回答的助手。遵守以下规则：\n"
        "1. 禁止直接回答任何问题，必须先调用相关工具获取真实数据\n"
        "2. 天气问题 → 调用 get_weather\n"
        "3. 股价问题 → 调用 get_stock\n"
        "4. 公司制度/福利/产品 → 调用 query_knowledge\n"
        "5. 员工/产品/订单数据 → 调用 execute_sql\n"
        "6. 翻译请求 → 调用 translate\n"
        "7. 计算题 → 调用 calculate\n"
        "8. 其他需要最新信息的 → 调用 search_web\n"
        "9. 读取文件内容 → 调用 read_file\n"
        "10. 工具返回数据后，基于真实数据组织回复\n"
        "11. 如果工具调用失败，如实告知用户失败原因"
    )

    # ── 请求控制 ──
    RETRIES     = int(os.getenv("AGENT_RETRIES", "2"))
    TIMEOUT     = int(os.getenv("TIMEOUT", "8"))
    MAX_ROUNDS  = int(os.getenv("MAX_ROUNDS", "3"))
    HISTORY_MAX = int(os.getenv("HISTORY_MAX", "6"))
    TOOL_RESULT_MAX = int(os.getenv("TOOL_RESULT_MAX", "500"))

    # ── 重试 & 熔断 ──
    TOOL_RETRIES     = int(os.getenv("TOOL_RETRIES", "2"))
    RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
    BREAKER_MAX_FAIL = int(os.getenv("BREAKER_MAX_FAIL", "3"))
    BREAKER_COOLDOWN = int(os.getenv("BREAKER_COOLDOWN", "30"))

    # ── 持久化记忆 ──
    MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes")
    MEMORY_SESSION = os.getenv("MEMORY_SESSION", "default")

    # ── API 服务 ──
    API_HOST = os.getenv("HOST", "0.0.0.0")
    API_PORT = int(os.getenv("PORT", "8000"))

    @classmethod
    def validate(cls) -> list[str]:
        """启动时校验配置，返回警告列表"""
        warnings = []
        if not cls.DEEPSEEK_KEY:
            warnings.append("DEEPSEEK_API_KEY 未设置！请在 .env 文件中填写密钥")
        elif not cls.DEEPSEEK_KEY.startswith("sk-"):
            warnings.append("DEEPSEEK_API_KEY 疑似无效（应以 sk- 开头）")
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 2:
            warnings.append(f"TEMPERATURE={cls.TEMPERATURE} 超出合理范围 0-2")
        if not cls.MODEL:
            warnings.append("MODEL 未设置")
        return warnings
