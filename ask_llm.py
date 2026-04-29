import ollama
import pandas as pd
from pathlib import Path


def classify_image(img_path:str, model:str, question:str, sys_prompt:str) -> str:
    # process an image
    with open(img_path, 'rb') as f:
        image_data = f.read()

    # ask LLM
    try:
        response = ollama.chat(
            model=model,
            messages=[
            {
                'role': 'system',
                'content': sys_prompt,
            },
            {
                'role': 'user',
                'content': question,
                'images': [image_data],
            },
            ],
            options={"temperature": 0.01}
        )
        answ = response['message']['content']
        return answ
    except:
        return "skip"
    

# An example of usage

# predefine all settings
images = [str(i) for i in list(Path("./ReIV/Re-Imagine Vision/Re-Imagine Vision Bench/original/cats_orig").rglob("*.jpg"))]
models = ["gemma3:12b","mistral-small3.1:latest","qwen2.5vl:7b","mistral-small3.2:24b","qwen3-vl:32b","qwen3-vl:30b",
          "blaifa/InternVL3_5:8b","ebdm/gemma3-enhanced:12b","internlm/interns1:mini","llama4:latest","gemma3:latest",
          "llama3.2-vision:latest","gemma3:27b"]

questions = ['How much do you think that the image is related to a CAT?', 'How much do you think that the image is related to a DOG?']
sys_prompt = "You are an image classifier. You have two possible classes: CAT and DOG. You should send ONLY an integer number from 0 to" \
            " 100 which would reflect how much do you think that the image is related to a CLASS"

# create dataframes for answers storage and the questions which need to be reasked
df = pd.DataFrame(columns=["model", "img_path", "target", "model_answ", "question"])
df_skipped = pd.DataFrame(columns=["model", "img_path"])

# get models results
for model in models:
    for img in images:
        for question in questions:
            answer = classify_image(img, model, question, sys_prompt)

            if answer == "skip":
                df_skipped.loc[len(df_skipped)] = [model, img]
                df_skipped.to_csv("cats_skipped.csv", index=False)
            else:
                df.loc[len(df)] = [model, img, "CAT", answer, question]
                df.to_csv("llms_cats_orig.csv", index=False)