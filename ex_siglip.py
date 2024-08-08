import torch
import requests
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
device = "cuda" # the device to load the model onto

model = SiglipModel.from_pretrained(
    "google/siglip-so400m-patch14-384",
    torch_dtype=torch.float16,
    device_map=device,
)
processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = [f"a humanoid robot kneeling"]
inputs = processor(text=candidate_labels, images=image, padding="max_length", return_tensors="pt")
inputs.to(device)

with torch.no_grad():
    with torch.autocast(device):
        outputs = model(**inputs)

similarity = outputs.logits_per_text

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")
