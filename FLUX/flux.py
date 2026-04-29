import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
import random
from pathlib import Path
from datetime import datetime
import os

repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda:0"
torch_dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
)
pipe.to(device)

images = [str(i) for i in list(Path("./dogs").rglob("*.jpg"))]

count = 0
for img_path in images:
    print(datetime.now())
    print(count)
    count += 1

    os.makedirs(f"./manipulated_dogs/{Path(img_path).stem}", exist_ok=True)

    color = random.choice(["purple", "green", "pink", "blue", "red"])

    prompts = {"dog without ears": "CF","add a butterfly (not on the dog, in the corner)": "UI","add a bone near the dog":"AD",f"change fur color to {color}":"SV","add a butterfly on the dog's nose":"IC"}

    for prompt in list(prompts.keys()):
        dog_image = load_image(img_path).convert("RGB")
        image = pipe(
            prompt=prompt,
            image=[dog_image],
            generator=torch.Generator(device=device).manual_seed(73),
            num_inference_steps=20,
            guidance_scale=4,
        ).images[0]
        out_path = f"./manipulated_dogs/{Path(img_path).stem}/" + Path(img_path).stem + f"_{prompts[prompt]}.jpg"
        image.save(out_path)
        print(out_path)