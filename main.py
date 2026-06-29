import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from src.api.routes import router
from src.api.errors import global_exception_handler

STATIC_DIR = os.path.join(os.path.dirname(__file__), "src", "static")

app = FastAPI(
    title="Personal Knowledge Base RAG",
    description="基于 RAG 技术的个人知识库智能问答系统",
    version="2.0.0",
)

# 全局异常处理
app.add_exception_handler(Exception, global_exception_handler)

# API 路由
app.include_router(router)

# 静态文件
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """返回前端 SPA"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Personal Knowledge Base RAG API", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
