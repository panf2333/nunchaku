from pydantic import BaseModel


class ImageResponse(BaseModel):
    url: str
    
class BaseResponse(BaseModel):
    code: int
    message: str
    data: list[ImageResponse]
