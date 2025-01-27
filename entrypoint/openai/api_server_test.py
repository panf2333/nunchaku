import asyncio
from io import BytesIO
import json
import logging
import os
import resource
import signal
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Optional
import uuid

import PIL
import psutil
import uvicorn
import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from logging import Logger
import torch
from diffusers import FluxPipeline

from create_image_request import CreateImageRequest
from base_response import BaseResponse
import s3_util  
# from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

VERSION = "1.0.0"
TIMEOUT_KEEP_ALIVE = 180  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = logging.getLogger('entrypoints.openai.api_server')
logger.setLevel(logging.INFO)

FORMAT = '%(asctime)s %(levelname)s %(message)s'
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(FORMAT))
logger.addHandler(console_handler)

pretrained_model_dict = {"mit-han-lab/svdq-int4-flux.1-dev":"black-forest-labs/FLUX.1-dev", "mit-han-lab/svdq-int4-flux.1-schnell":"black-forest-labs/FLUX.1-schnell"}
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("start server")
        yield
        
    finally:
        # Ensure app state including engine ref is gc'd
        torch.cuda.empty_cache()
        logger.info("stop server empty cuda")
        del app.state


router = APIRouter()



def init_app_state(app_state, args):
    app_state.model_name = args.model_name
    app_state.dtype = args.dtype
    config = read_config_json('config.json')
    app_state.s3_config = config["s3"]
    app_state.s3_client = s3_util.get_s3_client(app_state.s3_config)
    app_state.s3_bucket = app_state.s3_config['bucket']
    app_state.s3_prefix_path = app_state.s3_config["prefix_path"] + "/"

# def init_app_state(app_state, pipeline, args):
#     app_state.model_name = args.model_name
#     app_state.dtype = args.dtype
#     app_state.pipeline = pipeline


    # app_state.pipeline = pipeline
# def load_pipeline(args) -> FluxPipeline:
#     model_name = args.model_name
#     if args.model_name == "dev":
#         model_name = "mit-han-lab/svdq-int4-flux.1-dev"
#         dtype = torch.float16
#     elif args.model_name == "schnell":
#         model_name = "mit-han-lab/svdq-int4-flux.1-schnell"
#         dtype = torch.float16
#     else:
#        raise ValueError("Invalid model name")
#     pretrained_model_name_or_path = pretrained_model_dict[model_name]
#     transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_name)
#     pipeline = FluxPipeline.from_pretrained(
#     pretrained_model_name_or_path, transformer=transformer, torch_dtype=dtype).to("cuda")
#     return pipeline

@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)


@router.api_route("/v1/images/generations", methods=["GET", "POST"])
async def imagesGenerations(req: CreateImageRequest, raw_req: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    # image = PIL.Image.open("input_image.jpg")
    # # image = req.app.state.pipeline(req.prompt, req.num_inference_steps, req.guidance_scale).images[0]
    # path = f"output-{uuid.uuid4()}.png"
    # logger.info(f"Saving image to {path}")
    # print(f"Saving image to {path}")
    # image.save(path)

    # # 创建一个 BytesIO 对象
    # buf = BytesIO()
    # image.save(buf, format='JPEG')
    # buf.seek(0)

    # return Response(buf.read(), media_type="image/jpeg")
    # return Response(status_code=200)
    bucket = raw_req.app.state.s3_bucket
    object_name = raw_req.app.state.s3_prefix_path + f"output-{uuid.uuid4()}.png"
    file_name = "input_image.jpg"
    # Open the file in binary mode
    # Get s3_client from app state
    s3_client = raw_req.app.state.s3_client  
    try:
        with open(file_name, 'rb') as file:
            # Upload the file
            s3_util.upload_fileobj(s3_client, file, bucket, object_name)
            logger.info(f"File {file_name} uploaded to {bucket}/{object_name}")
            response_url = s3_util.create_presigned_url(s3_client, bucket, object_name)
            if response_url is not None:
                logger.info(f"Presigned URL: {response_url}")
                result = BaseResponse(code=10000, message="success", data=[{"url": response_url}])
            else:
                result = BaseResponse(code=10001, message="failed to generate presigned URL", data=[])
    except Exception as e:
        logger.error(f"Error uploading file {file_name}: {e}")
        result = BaseResponse(code=10001, message="failed to upload file", data=[])     
    return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.OK)

@router.get("/version")
async def show_version():
    version = {"version": VERSION}
    return JSONResponse(content=version)

def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        logger.error(exc)
        return JSONResponse(content="BAD_REQUEST", status_code=HTTPStatus.BAD_REQUEST)

    return app


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("nunchaku API server version %s", VERSION)
    logger.info("args: %s", args)


    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)
    app = build_app(args)
    # async with load_pipeline(args) as pipeline:

    #     init_app_state(app.state, pipeline, args)
    #     shutdown_task = await serve_http(
    #         app,
    #         host=args.host,
    #         port=args.port,
    #         timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    #         **uvicorn_kwargs,
    #     )

    init_app_state(app.state, args)
    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        **uvicorn_kwargs,
    )
    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()

def find_process_using_port(port: int) -> Optional[psutil.Process]:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # of vLLM. For now, let's not look for processes in this case.
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/
    if sys.platform.startswith("darwin"):
        return None

    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, __):

        pipeline = request.app.state.pipeline

        logger.fatal("RuntimeError, terminating server "
                         "process")
        server.should_exit = True
        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase"
                "with error %s. This can cause fd limit errors like"
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n", current_soft, e)

def read_config_json(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def mark_args(parser: ArgumentParser) -> None:
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--allowed-origins", type=list, default=["*"])
    parser.add_argument("--allow-credentials", type=bool, default=True)
    parser.add_argument("--allowed-methods", type=list, default=["*"])
    parser.add_argument("--allowed-headers", type=list, default=["*"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="schnell")
    mark_args(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
