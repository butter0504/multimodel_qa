from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core.config import settings
from api.routes import table, text, image


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(table.router, prefix=f"{settings.API_V1_STR}/table", tags=["table"])
app.include_router(text.router, prefix=f"{settings.API_V1_STR}/text", tags=["text"])
app.include_router(image.router, prefix=f"{settings.API_V1_STR}/image", tags=["image"])


@app.get("/")
def root():
    """根路径接口"""
    return {
        "message": "多模态数据质量检测系统 API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "table_detect": "/api/v1/table/detect",
            "text_detect": "/api/v1/text/detect",
            "image_detect": "/api/v1/image/detect"
        }
    }


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "healthy"}
