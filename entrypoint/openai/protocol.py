from pydantic import BaseModel

########################### request ########################
class CreateImageRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    lora_name: str = "None"
    lora_weight: float = 0.8
    seed: int = 0
    # sana
    pag_scale: float = 2.0

#### safe check request ####
# Define a request model for the /safety/check/prompt endpoint
class SafeCheckPromptRequest(BaseModel):
    prompt: str

########################### request ########################


########################### response ########################
class ImageResponse(BaseModel):
    url: str
    latency: float
    is_safe_prompt: bool
    
class BaseResponse(BaseModel):
    code: int
    message: str
    data: list[ImageResponse]

class ModelStatus(BaseModel):
    status: str
    model: str
    
class HealthCheckResponse(BaseModel):
    code: int
    message: str
    data: ModelStatus

# Define a response model for the /safety/check/prompt endpoint
class SafetyCheckResponse(BaseModel):
    prompt: str
    is_safe: bool

########################### response ########################


########################### config ########################
class S3Config(BaseModel):
    bucket: str
    prefix_path: str
    aws_access_key_id: str
    aws_secret_access_key: str

class Config(BaseModel):
    s3: S3Config
    safe_check_url: str
########################### config ########################
