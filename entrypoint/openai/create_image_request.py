from fastapi import Request

# https://huggingface.co/docs/diffusers/v0.32.2/en/api/pipelines/flux
# num_inference_steps (int, optional, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.  
class CreateImageRequest(Request):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    lora_weight: float = 0.8
    seed: int = 0