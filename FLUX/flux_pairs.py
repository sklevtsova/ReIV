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

# images = [str(i) for i in list(Path("./cat_and_dog").rglob("*.jpg"))]

# count = 0

# for img_path in images:
#     print(datetime.now())
#     print(count)
#     count += 1

#     os.makedirs(f"./manipulated_pairs/{Path(img_path).stem}", exist_ok=True)

#     prompts = ["Add a bottle with milk in front of, to the side of, or next to the cat and the dog. Do not replace animals with a bottle.",
#                 "Add a bone in front of, to the side of, or next to the cat and the dog. Do not replace animals with bones."]
#     for prompt in prompts:
#         cat_image = load_image(img_path).convert("RGB")
#         image = pipe(
#             prompt=prompt,
#             image=[cat_image],
#             generator=torch.Generator(device=device).manual_seed(42),
#             num_inference_steps=20,
#             guidance_scale=4,
#         ).images[0]

#         if prompt == prompts[0]:
#             out_path = f"./manipulated_pairs/{Path(img_path).stem}/" + Path(img_path).stem + "_milk.jpg"
#         elif prompt == prompts[1]:
#             out_path = f"./manipulated_pairs/{Path(img_path).stem}/" + Path(img_path).stem + "_bone.jpg"

#         image.save(out_path)
#         print(out_path)

bone = [1,12,14,30,36,41,45,50,69,83,86,129,135,136,144,145,156,158,186,188,192,201]
milk = [12,57,83,106,107,132,133,161,162,186,191,197,201,217,218,233,245,254,257,264,266,268]

images = [str(i) for i in list(Path("./cat_and_dog").rglob("*.jpg"))]

count = 0

for img_path in images:
    print(datetime.now())
    print(count)
    count += 1

    # add a bone

    os.makedirs(f"./manipulated_pairs/{Path(img_path).stem}", exist_ok=True)

    pair_image = load_image(img_path).convert("RGB")
    image = pipe(
        prompt="Add a bone in front of, to the side of, or next to the cat and the dog. Do not replace animals with bones.",
        image=[pair_image],
        generator=torch.Generator(device=device).manual_seed(100),
        num_inference_steps=20,
        guidance_scale=4,
    ).images[0]

    out_path = f"./manipulated_pairs/{Path(img_path).stem}/" + Path(img_path).stem + "_bone.jpg"

    image.save(out_path)
    print(out_path)

    # add a bottle of milk

    image = pipe(
        prompt="Add a bottle with milk in front of, to the side of, or next to the cat and the dog. Do not replace animals with a bottle.",
        image=[pair_image],
        generator=torch.Generator(device=device).manual_seed(100),
        num_inference_steps=20,
        guidance_scale=4,
    ).images[0]

    out_path = f"./manipulated_pairs/{Path(img_path).stem}/" + Path(img_path).stem + "_milk.jpg"

    image.save(out_path)
    print(out_path)