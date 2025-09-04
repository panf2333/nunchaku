import argparse
import logging
from entrypoint.openai.log import setup_logging
from typing import Dict

from fastapi import Request
import numpy as np

from .flux_pix2pix_pipeline import FluxPix2pixTurboPipeline
import torch
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel
from .vars import DEFAULT_SKETCH_GUIDANCE, MAX_SEED, STYLES
from PIL import Image

blank_image = Image.new("RGB", (1024, 1024), (255, 255, 255))

setup_logging()

logger = logging.getLogger(__name__)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "fp4", "bf16"], help="Which precisions to use"
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker")
    parser.add_argument("--count-use", action="store_true", help="Whether to count the number of uses")
    parser.add_argument("--gradio-root-path", type=str, default="")
    args = parser.parse_args()
    return 

def get_pipeline(args) -> FluxPix2pixTurboPipeline:
    if args.precision == "bf16":
        pipeline = FluxPix2pixTurboPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipeline = pipeline.to("cuda")
        pipeline.precision = "bf16"
        pipeline.load_control_module(
            "mit-han-lab/svdq-flux.1-schnell-pix2pix-turbo", "sketch.safetensors", alpha=DEFAULT_SKETCH_GUIDANCE
        )
    else:
        assert args.precision in ["int4", "fp4"]
        pipeline_init_kwargs = {}
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{args.precision}_r32-flux.1-schnell.safetensors"
        )
        if args.use_fp16_attention:
            # set attention implementation to fp16
            transformer.set_attention_impl("nunchaku-fp16")  
        pipeline_init_kwargs["transformer"] = transformer
        if args.use_qencoder:
            from nunchaku.models.text_encoders.t5_encoder import NunchakuT5EncoderModel

            text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
                "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
            )
            pipeline_init_kwargs["text_encoder_2"] = text_encoder_2

        pipeline = FluxPix2pixTurboPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
        pipeline = pipeline.to("cuda")
        pipeline.precision = args.precision
        pipeline.load_control_module(
            "mit-han-lab/svdq-flux.1-schnell-pix2pix-turbo",
            "sketch.safetensors",
            alpha=DEFAULT_SKETCH_GUIDANCE,
        )
    return pipeline

def generate_image(req, raw_req: Request, images: Dict[str, Image]) -> Image:
    pipeline = raw_req.app.state.pipeline
    prompt = req.prompt
    image = images["composite"]
    image_numpy = np.array(image.convert("RGB"))
    if prompt.strip() == "" and (np.sum(image_numpy == 255) >= 3145628 or np.sum(image_numpy == 0) >= 3145628):
        return blank_image, "Please input the prompt or draw something."
    
    prompt_template = STYLES[req.styles]
    prompt = prompt_template.format(prompt=prompt)
    # Validate req.seed
    if not (0 <= req.seed <= MAX_SEED):
        raise ValueError(f"Seed must be between 0 and {MAX_SEED}.")

    # Validate req.sketch_guidance
    if not (0 <= req.sketch_guidance <= 1):
        raise ValueError("Sketch guidance must be between 0 and 1.")
    # Validate step for sketch_guidance (0.01)
    # Using a small epsilon for float comparison due to potential precision issues
    if abs(req.sketch_guidance * 100 - round(req.sketch_guidance * 100)) > 1e-6:
        raise ValueError("Sketch guidance must be a multiple of 0.01.")
    logger.info(f"Prompt: {prompt}, alpha: {req.sketch_guidance}, seed: {req.seed}")
    return pipeline(
        image=image,
        image_type="sketch",
        alpha=req.sketch_guidance,
        prompt=prompt,
        generator=torch.Generator().manual_seed(req.seed),
    ).images[0]
