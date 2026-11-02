"""
电商客服 Agent · Web 服务
启动: .venv/Scripts/python.exe -m cs.server
访问: http://127.0.0.1:8001
"""
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cs.agent import CsAgent
from cs.config import CSConfig

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")

app = FastAPI(title="电商客服 Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话池：session_id -> CsAgent
_sessions: dict[str, CsAgent] = {}
_stats = {"total": 0, "handoff": 0, "ticket": 0}


def get_agent(sid: str) -> CsAgent:
    if sid not in _sessions:
        if len(_sessions) > 200:            # 简单防爆：超限清空最旧
            _sessions.pop(next(iter(_sessions)))
        _sessions[sid] = CsAgent(session_id=sid)
    return _sessions[sid]


# ── 请求/响应模型 ──

class ChatIn(BaseModel):
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default")


class TraceItem(BaseModel):
    tool: str
    args: dict
    result: str
    ms: int


class ChatOut(BaseModel):
    reply: str
    trace: list[TraceItem] = []
    rounds: int = 0
    handoff: bool = False
    tickets: list = []


class ResetIn(BaseModel):
    session_id: str = "default"


# ── 路由 ──

@app.get("/")
async def index():
    # 禁用缓存：前端改动后浏览器/预览面板常拿旧副本，导致「明明改了还是显示旧标题」
    return FileResponse(
        os.path.join(STATIC, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.post("/api/chat", response_model=ChatOut)
async def chat(req: ChatIn):
    if not req.message.strip():
        raise HTTPException(400, "消息不能为空")

    agent = get_agent(req.session_id)
    try:
        r = await agent.chat(req.message)
    except Exception as e:
        raise HTTPException(500, f"处理失败: {type(e).__name__}: {e}")

    _stats["total"] += 1
    if r["handoff"]:
        _stats["handoff"] += 1
    if r["tickets"]:
        _stats["ticket"] += len(r["tickets"])
    return ChatOut(**r)


@app.post("/api/reset")
async def reset(req: ResetIn):
    if req.session_id in _sessions:
        _sessions[req.session_id].reset()
    return {"status": "ok"}


@app.get("/api/stats")
async def stats():
    """自动解决率 = 未转人工 / 总对话。客服场景的核心业务指标"""
    total = _stats["total"] or 1
    return {
        "总对话数": _stats["total"],
        "转人工数": _stats["handoff"],
        "自动解决率": f"{(1 - _stats['handoff'] / total) * 100:.1f}%",
        "创建工单数": _stats["ticket"],
        "活跃会话": len(_sessions),
    }


@app.get("/api/health")
async def health():
    from cs.retrieval import get_backend, get_threshold

    return {
        "status": "ok",
        "model": CSConfig.MODEL,
        "has_key": bool(CSConfig.API_KEY),
        "retriever": get_backend(),
        "retriever_threshold": get_threshold(),
    }


def main():
    import uvicorn

    if not CSConfig.API_KEY:
        print("[ERROR] 未配置 DEEPSEEK_API_KEY，请检查 .env")
        return
    print("客服工作台 → http://127.0.0.1:8001")
    print("接口文档   → http://127.0.0.1:8001/docs")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")


if __name__ == "__main__":
    main()
