from fastapi import Form

# Assuming DEFAULT_SKETCH_GUIDANCE is defined in another file, import it
from sketch.vars import DEFAULT_SKETCH_GUIDANCE


class SketchToImageParams:
    """
    A dependency class to encapsulate all form parameters for the sketch-to-image endpoint.
    """

    def __init__(
        self,
        prompt: str = Form(...),
        sketch_guidance: float = Form(DEFAULT_SKETCH_GUIDANCE),
        num_inference_steps: int = Form(10),
        guidance_scale: float = Form(2.5),
        seed: int = Form(233),
        styles: str = Form("None"),
    ):
        self.prompt = prompt
        self.sketch_guidance = sketch_guidance
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.styles = styles

    # --- ADD THIS METHOD ---
    def __repr__(self) -> str:
        # This will dynamically create a string like "SketchToImageParams(prompt='...', seed=...)"
        # The !r in the f-string ensures that string values are correctly quoted.
        attrs = ", ".join(f"{key}={value!r}" for key, value in vars(self).items())
        return f"{self.__class__.__name__}({attrs})"
