import argparse
import logging
from typing import Dict

import torch
from diffusers import FluxFillPipeline
from fastapi import Request
from PIL import Image

from entrypoint.openai.log import setup_logging
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

from .vars import MAX_SEED, STYLES

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
    return args


def get_pipeline(args) -> FluxFillPipeline:
    if args.precision == "bf16":
        pipeline = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        pipeline = pipeline.to("cuda")
        pipeline.precision = "bf16"
    else:
        assert args.precision in ["int4", "fp4"]
        pipeline_init_kwargs = {}
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"mit-han-lab/nunchaku-flux.1-fill-dev/svdq-{args.precision}_r32-flux.1-fill-dev.safetensors"
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

        pipeline = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, **pipeline_init_kwargs
        )
        pipeline = pipeline.to("cuda")
        pipeline.precision = args.precision
    return pipeline


def generate_image(req, raw_req: Request, images: Dict[str, Image]) -> Image:
    pipeline = raw_req.app.state.pipeline

    prompt_template = STYLES[req.styles]
    prompt = prompt_template.format(prompt=req.prompt)

    # Validate req.seed
    if not (0 <= req.seed <= MAX_SEED):
        raise ValueError(f"Seed must be between 0 and {MAX_SEED}.")

    # Validate req.num_inference_steps
    if not (10 <= req.num_inference_steps <= 50):
        raise ValueError("Number of inference steps must be between 10 and 50.")

    # Validate req.guidance_scale
    if not (1 <= req.guidance_scale <= 50):
        raise ValueError("Guidance scale must be between 1 and 50.")
    # Validate step for guidance_scale (1)
    if abs(req.guidance_scale - round(req.guidance_scale)) > 1e-6:
        raise ValueError("Guidance scale must be a multiple of 1.")

    mask = images["layer_0"].getchannel(3)  # Mask is stored in the last channel
    pic = images["background"].convert("RGB")  # This is the original photo

    logger.info(
        f"Prompt: {prompt}, guidance_scale: {req.guidance_scale}, seed: {req.seed}, num_inference_steps: {req.num_inference_steps}"
    )
    return pipeline(
        prompt=prompt,
        image=pic,
        mask_image=mask,
        guidance_scale=req.guidance_scale,
        height=1024,
        width=1024,
        num_inference_steps=req.num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator().manual_seed(req.seed),
    ).images[0]
