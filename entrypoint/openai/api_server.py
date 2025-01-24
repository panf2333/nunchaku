import asyncio
import resource
import signal
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Optional

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

from createImageRequest import CreateImageRequest
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

VERSION = "1.0.0"
TIMEOUT_KEEP_ALIVE = 180  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = Logger('entrypoints.openai.api_server')

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

def init_app_state(app_state, pipeline, args):
    app_state.model_name = args.model_name
    app_state.dtype = args.dtype
    app_state.pipeline = pipeline

def load_pipeline(args) -> FluxPipeline:
    model_name = args.model_name
    dtype = args.dtype
    pretrained_model_name_or_path = pretrained_model_dict[model_name]
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_name)
    pipeline = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path, transformer=transformer, torch_dtype=dtype).to("cuda")
    return pipeline

@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)


@router.api_route("/v1/images/generations ", methods=["GET", "POST"])
async def imagesGenerations(req: CreateImageRequest) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    image = req.app.state.pipeline(req.prompt, req.num_inference_steps, req.guidance_scale).images[0]
    image.save("output.png")
    return Response(status_code=200)

@router.get("/version")
async def show_version():
    version = {"version": VERSION}
    return JSONResponse(content=version)

def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path


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
    async with load_pipeline(args) as pipeline:
        
        await init_app_state(app.state, pipeline, args)
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvloop.run(run_server(args))
