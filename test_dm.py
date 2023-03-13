from diffusers import StableDiffusionPipeline
import torch

model_id = "cabin_ti"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

# prompt = "<cabin>"
prompt = "A large <cabin> on top of a sunny mountain in the style of Dreamworks, artstation"


for i in range(20):
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

    image.save(f"test/cabin_new_{i}.png")
