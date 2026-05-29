from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    case_id: str | None = None


class ChatResponse(BaseModel):
    case_id: str
    message: str
    phase: str
    is_closed: bool
