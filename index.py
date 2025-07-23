from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
prompt = "southern lights"
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]
image.save("cyberpunk_city.png")
image.show()
display(image)