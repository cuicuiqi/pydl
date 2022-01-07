#!/usr/bin/env python
# coding: utf-8

# Uncomment to run on cpu
# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import random

import gradio as gr
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image, ImageDraw, ImageFont

# ## CLIP Scoring
from transformers import BartTokenizer, CLIPProcessor, FlaxCLIPModel
from vqgan_jax.modeling_flax_vqgan import VQModel

from dalle_mini.model import CustomFlaxBartForConditionalGeneration

DALLE_REPO = "flax-community/dalle-mini"
DALLE_COMMIT_ID = "4d34126d0df8bc4a692ae933e3b902a1fa8b6114"

VQGAN_REPO = "flax-community/vqgan_f16_16384"
VQGAN_COMMIT_ID = "90cc46addd2dd8f5be21586a9a23e1b95aa506a9"

tokenizer = BartTokenizer.from_pretrained(DALLE_REPO, revision=DALLE_COMMIT_ID)
model = CustomFlaxBartForConditionalGeneration.from_pretrained(
    DALLE_REPO, revision=DALLE_COMMIT_ID
)
vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)


def captioned_strip(images, caption=None, rows=1):
    increased_h = 0 if caption is None else 48
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images) * w // rows, h * rows + increased_h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i // rows * w, increased_h + (i % rows) * h))

    if caption is not None:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 40
        )
        draw.text((20, 3), caption, (255, 255, 255), font=font)
    return img


def custom_to_pil(x):
    x = np.clip(x, 0.0, 1.0)
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def generate(input, rng, params):
    return model.generate(
        **input,
        max_length=257,
        num_beams=1,
        do_sample=True,
        prng_key=rng,
        eos_token_id=50000,
        pad_token_id=50000,
        params=params,
    )


def get_images(indices, params):
    return vqgan.decode_code(indices, params=params)


p_generate = jax.pmap(generate, "batch")
p_get_images = jax.pmap(get_images, "batch")

bart_params = replicate(model.params)
vqgan_params = replicate(vqgan.params)

clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("Initialize FlaxCLIPModel")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Initialize CLIPProcessor")


def hallucinate(prompt, num_images=64):
    prompt = [prompt] * jax.device_count()
    inputs = tokenizer(
        prompt,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=128,
    ).data
    inputs = shard(inputs)

    all_images = []
    for i in range(num_images // jax.device_count()):
        key = random.randint(0, 1e7)
        rng = jax.random.PRNGKey(key)
        rngs = jax.random.split(rng, jax.local_device_count())
        indices = p_generate(inputs, rngs, bart_params).sequences
        indices = indices[:, :, 1:]

        images = p_get_images(indices, vqgan_params)
        images = np.squeeze(np.asarray(images), 1)
        for image in images:
            all_images.append(custom_to_pil(image))
    return all_images


def clip_top_k(prompt, images, k=8):
    inputs = processor(text=prompt, images=images, return_tensors="np", padding=True)
    outputs = clip(**inputs)
    logits = outputs.logits_per_text
    scores = np.array(logits[0]).argsort()[-k:][::-1]
    return [images[score] for score in scores]


def compose_predictions(images, caption=None):
    increased_h = 0 if caption is None else 48
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images) * w, h + increased_h))
    for i, img_ in enumerate(images):
        img.paste(img_, (i * w, increased_h))

    if caption is not None:
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 40
        )
        draw.text((20, 3), caption, (255, 255, 255), font=font)
    return img


def top_k_predictions(prompt, num_candidates=32, k=8):
    images = hallucinate(prompt, num_images=num_candidates)
    images = clip_top_k(prompt, images, k=k)
    return images


def run_inference(prompt, num_images=32, num_preds=8):
    images = top_k_predictions(prompt, num_candidates=num_images, k=num_preds)
    predictions = captioned_strip(images)
    output_title = f"""
    <b>{prompt}</b>
    """
    return (output_title, predictions)


outputs = [
    gr.outputs.HTML(label=""),  # To be used as title
    gr.outputs.Image(label=""),
]

description = """
DALL·E-mini is an AI model that generates images from any prompt you give! Generate images from text:
"""
gr.Interface(
    run_inference,
    inputs=[gr.inputs.Textbox(label="What do you want to see?")],
    outputs=outputs,
    title="DALL·E mini",
    description=description,
    article="<p style='text-align: center'> Created by Boris Dayma et al. 2021 | <a href='https://github.com/borisdayma/dalle-mini'>GitHub</a> | <a href='https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA'>Report</a></p>",
    layout="vertical",
    theme="huggingface",
    examples=[
        ["an armchair in the shape of an avocado"],
        ["snowy mountains by the sea"],
    ],
    allow_flagging=False,
    live=False,
    # server_port=8999
).launch(share=True)
