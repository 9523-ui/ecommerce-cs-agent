"""
工具基类 —— pydantic 自动生成 schema + async 接口
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel


class BaseTool(ABC):
    """
    所有工具的抽象基类。
    子类只需：
      1) 定义 Params 内类 (pydantic) → schema 自动生成
      2) 设置 name / description
      3) 实现 async execute(**kwargs) → str
    """

    class Params(BaseModel):
        """参数模型 —— 子类覆写，用 Field(description=...) 声明"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（OpenAI function name）"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（AI 据此决定是否调用）"""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """执行工具（异步）"""
        ...

    def to_openai_schema(self) -> dict:
        """自动生成 OpenAI Function Calling schema"""
        js = self.Params.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": js.get("properties", {}),
                    "required": js.get("required", []),
                },
            },
        }
