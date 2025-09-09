MAX_IMAGE_SIZE = 2048
MAX_SEED = 1000000000

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024


MODEL_MAPPINGS = {
    "schnell": {
        "fp4": "mit-han-lab/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell",
        "int4": "mit-han-lab/svdq-int4-flux.1-schnell",
        "bf16": "black-forest-labs/FLUX.1-schnell",
    },
    "dev": {"int4": "mit-han-lab/svdq-int4-flux.1-dev", "bf16": "black-forest-labs/FLUX.1-dev"},
    "sana": {
        "int4": "mit-han-lab/svdq-int4-sana-1600m",
        "bf16": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    },
    "schnell_sketch": {"fp4": "mit-han-lab/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell-sketch"},
    "kontext": {"fp4": "mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-fp4_r32-flux.1-kontext-dev"},
    "fill": {"fp4": "mit-han-lab/nunchaku-flux.1-fill-dev/svdq-fp4_r32-flux.1-fill-dev"},
    "depth": {"fp4": "mit-han-lab/nunchaku-flux.1-depth-dev/svdq-fp4_r32-flux.1-depth-dev"},
}
