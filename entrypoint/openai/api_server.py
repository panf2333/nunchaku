import asyncio
from io import BytesIO
import json
import logging
import resource
import signal
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
import time
from typing import Any, Optional
import uuid

import psutil
import uvicorn
import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import torch
from create_image_request import CreateImageRequest
from base_response import BaseResponse, ImageResponse
from entrypoint import load_pipeline
from entrypoint.openai.log import setup_logging
from entrypoint.vars import PROMPT_TEMPLATES
from nunchaku.models.safety_checker import SafetyChecker
import s3_util  

VERSION = "1.0.0"
TIMEOUT_KEEP_ALIVE = 180  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = setup_logging()

router = APIRouter()

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

@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)

@router.get("/version")
async def show_version():
    version = {"version": VERSION}
    return JSONResponse(content=version)

@router.api_route("/v1/images/generations", methods=["GET", "POST"])
async def imagesGenerations(req: CreateImageRequest, raw_req: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    state = raw_req.app.state
    prompt = req.prompt
    logger.info(f"req: {req}")
    try:
        lora_name = state.lora_name
        if not state.safety_checker(prompt):
            prompt = "A peaceful world."
            logger.info("Unsafe prompt detected")
        prompt = PROMPT_TEMPLATES[lora_name].format(prompt=prompt)
        start_time = time.time()
        image = raw_req.app.state.pipeline(prompt, num_inference_steps=req.num_inference_steps, guidance_scale=req.guidance_scale).images[0]
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"start_time: {start_time}, end_time: {end_time}, latency: {latency}")
    except Exception as e:
        logger.exception("imagesGenerations failed")
        result = BaseResponse(code=10001, message="failed to generation image", data=[])    
        return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    
    bucket = state.s3_bucket
    object_name = state.s3_prefix_path + f"{state.model}-{state.precision}-{uuid.uuid4()}.png"
    s3_client = state.s3_client
    
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    url = s3_util.upload_file_and_get_presigned_url(s3_client, bucket, object_name, image_bytes)
    if url is not None:
        image_response = ImageResponse(url=url, latency=latency)
        result = BaseResponse(code=10000, message="success", data=[image_response])
    else:
        result = BaseResponse(code=10001, message="failed to generation image", data=[])    
    return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.OK)

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
    pipeline = load_pipeline.get_pipeline(args.model, args.precision, args.use_qencoder, args.lora_name, args.lora_weight)
    logger.info("Loaded pipeline")
    init_app_state(app.state, pipeline, args)
    logger.info("Initialized app state")
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

def init_app_state(app_state, pipeline, args):
    app_state.model = args.model
    app_state.precision = args.precision
    app_state.pipeline = pipeline
    app_state.lora_name = args.lora_name
    logger.info("read config.json")
    config = read_config_json('config.json')
    logger.info("get config done")
    app_state.s3_config = config["s3"]
    app_state.s3_client = s3_util.get_s3_client(app_state.s3_config)
    app_state.s3_bucket = app_state.s3_config['bucket']
    app_state.s3_prefix_path = app_state.s3_config["prefix_path"] + "/"
    logger.info(f"start init safety checker {args.no_safety_checker}")
    app_state.safety_checker = SafetyChecker("cuda", disabled=args.no_safety_checker)
    logger.info("end init safety checker")

def mark_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-m", "--model", type=str, default="schnell", choices=["schnell", "dev"], help="Which FLUX.1 model to use"
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="int4",
        nargs=1,
        choices=["int4", "bf16"],
        help="Which precisions to use",
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder", default=False)
    parser.add_argument("--lora-name", default="None", choices=["None", "All", "Anime", "GHIBSKY Illustration", "Realism", "Yarn Art", "Children Sketch"])
    parser.add_argument("--lora-weight", type=float, default=1.0)
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker", default=False)

    parser.add_argument("--allowed-origins", type=list, default=["*"])
    parser.add_argument("--allow-credentials", type=bool, default=True)
    parser.add_argument("--allowed-methods", type=list, default=["*"])
    parser.add_argument("--allowed-headers", type=list, default=["*"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    mark_args(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
