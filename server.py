from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import math
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import Counter, OrderedDict

from vocab import idx2word, word2idx

# ----------------- FastAPI Setup -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DEVICE -----------------
device = torch.device("cpu")  # CPU only

# ----------------- FEATURE EXTRACTOR -----------------
try:
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
except Exception as e:
    print(f"Warning: Could not load default weights, using untrained model: {e}")
    resnet_model = models.resnet50(weights=None)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # remove classifier
resnet_model.eval()
resnet_model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

def extract_features(image: Image.Image) -> torch.Tensor:
    img_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        features = resnet_model(img_tensor).view(1, -1)  # flatten
    return features  # shape [1, 2048]

# ----------------- LSTM CAPTION MODEL -----------------
class myModel(nn.Module):

    def __init__(self,vocab_size):

        super().__init__()
        self.hidden_size=512
        self.input_size=512

        self.encoder=nn.Sequential(
            nn.Linear(2048,self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        #self.encoder=nn.Linear(2048,self.hidden_size)
        self.embedding=nn.Embedding(vocab_size,self.input_size)
        self.lstm=nn.LSTM(self.input_size,self.hidden_size,batch_first=True,num_layers=1,dropout=0.3)
        self.finallayer=nn.Linear(512,vocab_size)

    def forward(self,image_features,caption):

        h0=self.encoder(image_features)
        h0=h0.unsqueeze(0)
        c0=torch.zeros_like(h0)

        x=self.embedding(caption)

        intermediate_hidden_states, (final_hidden_state, final_cell_state)=self.lstm(x,(h0,c0))
        output=self.finallayer(intermediate_hidden_states)
        return output
# ----------------- LOAD PRETRAINED MODEL -----------------
#vocab_size = 20061  model3.pth
vocab_size = 20279 #model5.pth

model = myModel(vocab_size=vocab_size).to(device)

checkpoint = torch.load("model5.pth", map_location=device)
# Remove DataParallel prefix if exists
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

# ----------------- VOCAB -----------------
# Replace with your trained vocab

# ----------------- CAPTION GENERATION -----------------
def generate_caption(features: torch.Tensor, max_len=20):
    """Greedy search caption generation"""
    input_word = torch.tensor([[word2idx["<start>"]]], dtype=torch.long).to(device)
    caption = []

    h0 = model.encoder(features).unsqueeze(0)
    c0 = torch.zeros_like(h0)
    hidden = (h0, c0)

    for _ in range(max_len):
        x_embed = model.embedding(input_word)
        _, hidden = model.lstm(x_embed, hidden)
        output = model.finallayer(hidden[0].squeeze(0))
        predicted_idx = output.argmax(1).item()
        word = idx2word.get(predicted_idx, "<unk>")

        if word == "<end>":
            break
        caption.append(word)
        input_word = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)

    return " ".join(caption)


def generate_caption_beam_search(features: torch.Tensor, max_len=20, beam_width=3):
    """Beam search caption generation"""
    h0 = model.encoder(features).unsqueeze(0)
    c0 = torch.zeros_like(h0)

    # Each element: (score, word_indices, hidden_state, cell_state)
    beams = [(0.0, [word2idx["<start>"]], h0, c0)]
    complete_captions = []

    for _ in range(max_len):
        candidates = []

        for score, word_indices, hidden, cell in beams:
            if word_indices and idx2word.get(word_indices[-1]) == "<end>":
                complete_captions.append((score, word_indices))
                continue

            last_word = torch.tensor([[word_indices[-1]]], dtype=torch.long).to(device)
            x_embed = model.embedding(last_word)

            _, (new_hidden, new_cell) = model.lstm(x_embed, (hidden, cell))
            output = model.finallayer(new_hidden.squeeze(0))
            log_probs = torch.nn.functional.log_softmax(output, dim=-1)

            top_log_probs, top_indices = torch.topk(
                log_probs[0], min(beam_width, log_probs.size(1))
            )

            for log_prob, idx in zip(top_log_probs, top_indices):
                idx_item = idx.item()
                new_score = score + log_prob.item()
                new_word_indices = word_indices + [idx_item]
                candidates.append((new_score, new_word_indices, new_hidden, new_cell))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_width]

    complete_captions.extend((score, words) for score, words, _, _ in beams)
    if complete_captions:
        best_score, best_indices = max(complete_captions, key=lambda item: item[0])
    else:
        best_indices = []

    caption_words = []
    for idx in best_indices[1:]:
        word = idx2word.get(idx, "<unk>")
        if word == "<end>":
            break
        caption_words.append(word)

    return " ".join(caption_words)


def _normalize_tokens(text: str) -> list[str]:
    tokens = [t for t in text.lower().split() if t]
    return [t for t in tokens if t not in {"<start>", "<end>", "<pad>"}]


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    if n <= 0:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _precision_recall_f1(pred_tokens: list[str], ref_tokens: list[str], n: int) -> dict:
    pred_ngrams = _ngram_counts(pred_tokens, n)
    ref_ngrams = _ngram_counts(ref_tokens, n)
    overlap = sum((pred_ngrams & ref_ngrams).values())
    pred_total = sum(pred_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _bleu4(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        pred_ngrams = _ngram_counts(pred_tokens, n)
        ref_ngrams = _ngram_counts(ref_tokens, n)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        # Add-one smoothing to avoid zero precision.
        p_n = (overlap + 1) / (total + 1) if total else 0.0
        precisions.append(p_n)

    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / pred_len))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
    return score

# ----------------- FASTAPI ENDPOINT -----------------
@app.post("/predict")
async def predict_caption(
    file: UploadFile = File(...),
    reference: str | None = Form(default=None),
    references: str | None = Form(default=None),
    search_method: str = Form(default="greedy"),
    beam_width: int = Form(default=3),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Extract feature vector
    features = extract_features(image)

    # Generate caption based on search method
    search_key = search_method.lower().replace(" ", "")
    if search_key in {"beam", "beamsearch", "beam_search"}:
        caption = generate_caption_beam_search(features, beam_width=beam_width)
    else:
        caption = generate_caption(features)

    if not reference and not references:
        return {"caption": caption}

    if references:
        try:
            ref_list = json.loads(references)
        except json.JSONDecodeError:
            ref_list = []
        if not isinstance(ref_list, list):
            ref_list = []
    else:
        ref_list = [reference]

    pred_tokens = _normalize_tokens(caption)

    best = None
    for ref in ref_list:
        if not isinstance(ref, str) or not ref.strip():
            continue
        ref_tokens = _normalize_tokens(ref)
        token_metrics = _precision_recall_f1(pred_tokens, ref_tokens, n=1)
        ngram_metrics = _precision_recall_f1(pred_tokens, ref_tokens, n=4)
        bleu4 = _bleu4(pred_tokens, ref_tokens)
        candidate = {
            "bleu4": bleu4,
            "token_level": token_metrics,
            "ngram_level": {"n": 4, **ngram_metrics},
        }
        if best is None or candidate["bleu4"] > best["bleu4"]:
            best = candidate

    if best is None:
        return {"caption": caption}

    return {
        "caption": caption,
        "metrics": best,
    }
