import torch
from diffusers import Step1XEditPipelineV1P2
from diffusers.utils import load_image
from pathlib import Path
import os
from datetime import datetime
import random

pipe = Step1XEditPipelineV1P2.from_pretrained("stepfun-ai/Step1X-Edit-v1p2", torch_dtype=torch.bfloat16)
pipe.to("cuda")

images = [str(i) for i in list(Path("./cats").rglob("*.jpg"))]
seed = 100

count = 0
for img_path in images:
    print(datetime.now())
    print(count)
    count += 1

    image = load_image(img_path).convert("RGB")

    prompts = ["add an animal mouse near the cat"]

    os.makedirs(f"./manipulated_cats/{Path(img_path).stem}", exist_ok=True)


    for prompt in prompts:
        pipe_output = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=5,
            true_cfg_scale=6,
            generator=torch.Generator().manual_seed(seed)
        )

        if prompt == prompts[0]:
            out_path = f"./manipulated_cats/{Path(img_path).stem}/" + Path(img_path).stem + "_AD.jpg"

        pipe_output.final_images[0].save(out_path, lossless=True)
        print(out_path)
