import bentoml
from transformers import AutoModelWithLMHead, AutoTokenizer
from service import TransformerService
import torch

ts = TransformerService()
device = torch.device('cpu')

model = AutoModelWithLMHead.from_pretrained("hyunwoongko/kobart")
tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart")

model.load_state_dict(torch.load("sd2.pt", map_location=device))

# Option 1: Pack using dictionary (recommended)
artifact = {"model": model, "tokenizer": tokenizer}
ts.pack("BartModel", artifact)
saved_path = ts.save()
