import asyncio
import io
import logging
import os
import resource
import signal
import sys
import tempfile
import time
import uuid
from argparse import ArgumentParser, Namespace
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BytesIO
from typing import Any, List, Optional

import psutil
import torch
import uvicorn
import uvloop
from dependencies import SketchToImageParams
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from flux_utils import generate_i2i_image, generate_t2i_image, get_pipeline
from PIL import Image

from entrypoint.openai import s3_util
from entrypoint.openai.log import setup_logging
from entrypoint.openai.protocol import (
    BaseResponse,
    Config,
    CreateImageRequest,
    HealthCheckResponse,
    ImageResponse,
    ModelStatus,
    S3Config,
)
from entrypoint.vars import MODEL_MAPPINGS
from nunchaku.models.safety_checker import SafetyChecker

VERSION = "1.0.0"
TIMEOUT_KEEP_ALIVE = 180  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
setup_logging()

logger = logging.getLogger(__name__)

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
    state = raw_request.app.state
    model_status = ModelStatus(model=state.model_name, status="ok")
    result = HealthCheckResponse(code=10000, message="success", data=model_status)
    logger.info(f"Health check response {result.model_dump()}")
    return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.OK)


@router.get("/version")
async def show_version():
    version = {"version": VERSION}
    return JSONResponse(content=version)


@router.api_route("/v1/images/generations", methods=["GET", "POST"])
async def imagesGenerations(req: CreateImageRequest, raw_req: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    state = raw_req.app.state
    prompt = req.prompt
    is_safe_prompt = True
    logger.info(f"req: {req}")
    try:
        if not state.safety_checker(prompt):
            prompt = "A peaceful world."
            is_safe_prompt = False
            logger.info("Unsafe prompt detected")
        start_time = time.time()
        image = generate_t2i_image(req=req, raw_req=raw_req, prompt=prompt)
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"start_time: {start_time}, end_time: {end_time}, latency: {latency}")

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
    except Exception as e:
        logger.exception(f"imagesGenerations failed: {e}")
        result = BaseResponse(code=10001, message="failed to generation image", data=[])
        return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        del image
        torch.cuda.empty_cache()

    s3Config: S3Config = state.config.s3
    object_name = s3Config.prefix_path + f"{state.model}-{state.precision}-{uuid.uuid4()}.png"
    s3_client = state.s3_client

    url = s3_util.upload_file_and_get_presigned_url(s3_client, s3Config.bucket, object_name, image_bytes)
    if url is not None:
        image_response = ImageResponse(url=url, latency=latency, is_safe_prompt=is_safe_prompt)
        result = BaseResponse(code=10000, message="success", data=[image_response])
    else:
        result = BaseResponse(code=10001, message="failed to generation image", data=[])
    return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.OK)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ NEW ENDPOINT FOR SKETCH-TO-IMAGE BASED ON GRADIO 'RUN'
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@router.post("/v1/images/edits")
async def image_edits(
    raw_req: Request,
    # This is the main change: accepting a list of UploadFile objects
    images: List[UploadFile] = File(...),
    # The endpoint now looks even cleaner!
    req: SketchToImageParams = Depends(),
) -> Response:
    """
    Generates an image from a sketch and a text prompt.
    This endpoint accepts multipart/form-data.
    """
    state = raw_req.app.state
    logger.info(f"Received image edit request with {len(images)} images.")
    logger.info(f"Request parameters: {req}")  # Python automatically calls req.__repr__()

    try:
        # 2. Safety check for the prompt
        is_safe_prompt = True
        # Access prompt directly from req object
        prompt_for_safety_check = req.prompt
        if not state.safety_checker(prompt_for_safety_check):
            req.prompt = "A peaceful world."  # Modify req.prompt directly
            is_safe_prompt = False
            logger.info("Unsafe prompt detected, using default.")

        input_images = {}
        for image in images:
            # Get the filename to differentiate the images
            image_name_with_extension = image.filename
            image_name = os.path.splitext(image_name_with_extension)[0]  # Remove extension
            logger.info(f"Processing file: {image_name}")

            # Read file content and convert to a PIL Image
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents))

            # Store the image in the dictionary
            input_images[image_name] = pil_image

        # --- Using the differentiated images ---

        # 3. Run the pipeline (core logic from Gradio's run function)
        start_time = time.time()
        result_image = generate_i2i_image(req=req, raw_req=raw_req, images=input_images)
        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"Image generation latency: {latency:.4f}s")

        # 4. Convert result to bytes for response/upload
        image_bytes = BytesIO()
        result_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

    except Exception as e:
        logger.exception(f"images edits failed: {e}")
        result = BaseResponse(code=10001, message="Failed to generate image", data=[])
        return JSONResponse(content=result.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        # 5. Clean up CUDA memory
        torch.cuda.empty_cache()

    # 6. Upload to S3 and generate presigned URL
    s3Config: S3Config = state.config.s3
    object_name = s3Config.prefix_path + f"edit-{state.model}-{state.precision}-{uuid.uuid4()}.png"
    s3_client = state.s3_client

    url = s3_util.upload_file_and_get_presigned_url(s3_client, s3Config.bucket, object_name, image_bytes)

    if url is not None:
        image_response = ImageResponse(url=url, latency=latency, is_safe_prompt=is_safe_prompt)
        result = BaseResponse(code=10000, message="success", data=[image_response])
        status_code = HTTPStatus.OK
    else:
        result = BaseResponse(code=10001, message="Failed to upload generated image", data=[])
        status_code = HTTPStatus.INTERNAL_SERVER_ERROR

    return JSONResponse(content=result.model_dump(), status_code=status_code)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ END OF NEW ENDPOINT
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
    pipeline, processor = get_pipeline(args=args)

    logger.info("Loaded pipeline")
    app.state.pipeline = pipeline
    app.state.processor = processor
    init_app_state(app.state, args)
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

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    config = uvicorn.Config(app, log_config=None, **uvicorn_kwargs)
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
                "port %s is used by process %s launched with command:\n%s", port, process, " ".join(process.cmdline())
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # For now, let's not look for processes in this case.
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
        logger.fatal("RuntimeError, terminating server " "process")
        server.should_exit = True
        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase"
                "with error %s. This can cause fd limit errors like"
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def read_config():
    load_dotenv()
    bucket = os.getenv("S3_BUCKET")
    prefix_path = os.getenv("S3_PREFIX_PATH")
    aws_access_key_id = os.getenv("S3_AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
    safe_check_url = os.getenv("SAFE_CHECK_URL")

    # Non-empty checks for environment variables
    if not bucket:
        raise ValueError("S3_BUCKET environment variable must be set and not empty.")
    if not prefix_path:
        raise ValueError("S3_PREFIX_PATH environment variable must be set and not empty.")
    if not aws_access_key_id:
        raise ValueError("S3_AWS_ACCESS_KEY_ID environment variable must be set and not empty.")
    if not aws_secret_access_key:
        raise ValueError("S3_AWS_SECRET_ACCESS_KEY environment variable must be set and not empty.")

    s3 = S3Config(
        bucket=bucket,
        prefix_path=prefix_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    config = Config(s3=s3, safe_check_url=safe_check_url)
    return config


def init_app_state(app_state, args):
    app_state.model = args.model
    app_state.precision = args.precision
    app_state.model_name = MODEL_MAPPINGS[app_state.model][app_state.precision]
    app_state.lora_name = args.lora_name
    logger.info("load config")
    app_state.config = read_config()
    logger.info("get config done")
    app_state.s3_client = s3_util.get_s3_client(app_state.config.s3)
    logger.info(f"start init safety checker {args.no_safety_checker}")
    app_state.safety_checker = SafetyChecker(
        device="cuda", url=app_state.config.safe_check_url, disabled=args.no_safety_checker
    )
    logger.info("end init safety checker")


def mark_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="schnell",
        choices=["schnell", "dev", "sana", "schnell_sketch", "kontext", "fill", "canny", "depth"],
        help="Which model to use",
    )
    parser.add_argument(
        "-p", "--precision", type=str, default="int4", choices=["int4", "fp4", "bf16"], help="Which precisions to use"
    )

    parser.add_argument(
        "--use-fp16-attention", action="store_true", help="Whether to use nunchaku fp16 attention", default=False
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker")
    parser.add_argument("--count-use", action="store_true", help="Whether to count the number of uses")
    parser.add_argument(
        "--lora-name",
        default="All",
        choices=["None", "All", "Anime", "GHIBSKY Illustration", "Realism", "Yarn Art", "Children Sketch"],
    )
    parser.add_argument("--lora-weight", type=float, default=1.0)

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
