from inference_e1_1 import main, init_models
import random
from pathlib import Path
from datetime import datetime
import os


init_models()

images = [str(i) for i in list(Path("./dogs").rglob("*.jpg"))]

count = 0

for img_path in images:
    print(datetime.now())
    print(count)
    count += 1

    os.makedirs(f"./manipulated_dogs/HiDream/{Path(img_path).stem}", exist_ok=True)

    color = random.choice(["purple", "green", "pink", "blue", "red"])
    prompts = ["a dog without ears", "add a butterfly (not on the dog, in the corner)", f"change fur color to {color}", "add a butterfly on the dog's nose", "add a bone near the dog"]

    for prompt in prompts:

        if prompt == prompts[0]:
            out_path = f"./manipulated_dogs/HiDream/{Path(img_path).stem}/" + Path(img_path).stem + "_CF.jpg"
        elif prompt == prompts[1]:
            out_path = f"./manipulated_dogs/HiDream/{Path(img_path).stem}/" + Path(img_path).stem + "_UI.jpg"
        elif prompt == prompts[2]:
            out_path = f"./manipulated_dogs/HiDream/{Path(img_path).stem}/" + Path(img_path).stem + f"_SV_{color}.jpg"
        elif prompt == prompts[3]:
            out_path = f"./manipulated_dogs/HiDream/{Path(img_path).stem}/" + Path(img_path).stem + "_IC.jpg"
        elif prompt == prompts[4]:
            out_path = f"./manipulated_dogs/HiDream/{Path(img_path).stem}/" + Path(img_path).stem + "_AD.jpg"

        main(img_path, out_path, prompt)
        print(out_path)
    