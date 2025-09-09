import logging
from argparse import ArgumentParser
from typing import Dict

import depth_canny.utils as depth_canny_utils
import fill.utils as fill_utils
import kontext.utils as kontext_utils
import sketch.utils as sketch_utils
import t2i.utils as flux_t2i_utils
from fastapi import Request
from PIL.Image import Image

logger = logging.getLogger(__name__)


def get_pipeline(
    args: ArgumentParser,
):
    model_name = args.model
    precision = args.precision
    use_qencoder = args.use_qencoder
    lora_name = args.lora_name
    lora_weight = args.lora_weight
    use_fp16_attention = args.use_fp16_attention
    pipeline_init_kwargs: dict = {}
    processor = None
    if model_name in ["schnell", "schnell_v2", "dev"]:
        pipeline = flux_t2i_utils.get_pipeline(
            model_name=model_name,
            precision=precision,
            use_qencoder=use_qencoder,
            use_fp16_attention=use_fp16_attention,
            lora_name=lora_name,
            lora_weight=lora_weight,
            device="cuda",
            pipeline_init_kwargs=pipeline_init_kwargs,
        )
        pipeline.cur_lora_name = "None"
        pipeline.cur_lora_weight = 0
    elif model_name == "schnell_sketch":
        pipeline = sketch_utils.get_pipeline(args)
    elif model_name == "kontext":
        pipeline = kontext_utils.get_pipeline(args)
    elif model_name == "fill":
        pipeline = fill_utils.get_pipeline(args)
    elif model_name in ["depth", "canny"]:
        pipeline, processor = depth_canny_utils.get_pipeline(args)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return pipeline, processor


def generate_t2i_image(req, raw_req: Request, prompt: str) -> Image:
    return flux_t2i_utils.generate_image(req, raw_req, prompt)


def generate_i2i_image(req, raw_req: Request, images: Dict[str, Image]) -> Image:
    model = raw_req.app.state.model
    result_img = None
    if model == "schnell_sketch":
        result_img = sketch_utils.generate_image(req, raw_req, images)
    elif model == "kontext":
        result_img = kontext_utils.generate_image(req, raw_req, images)
    elif model == "fill":
        result_img = fill_utils.generate_image(req, raw_req, images)
    elif model in ["depth", "canny"]:
        result_img = depth_canny_utils.generate_image(req, raw_req, images)
    else:
        raise NotImplementedError(f"Model {model} not implemented")
    return result_img
