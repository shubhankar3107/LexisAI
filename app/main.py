from fastapi import FastAPI

from app.api.routes.uploads import router as uploads_router
from app.api.routes.documents import router as documents_router


app = FastAPI(
    title="LexisAI",
)

app.include_router(uploads_router)
app.include_router(documents_router)
