import torch
from torchvision import transforms
from PIL import Image
from src.model import build_model

# ── Config ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "models/vgg16_bn_classifier_vgg16_bn_best.pth"
SIZE = (224, 224)
LABELS = {0: "Not dishwasher-safe", 1: "Dishwasher-safe"}

# ── Load model once at startup ────────────────────────────────────────────
# torch.load on a state_dict checkpoint returns an OrderedDict, not a model.
# We need to rebuild the architecture first, then load the weights into it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model("vgg16_bn", num_classes=2, freeze_backbone=True)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint)

model.to(device)
model.eval()   # set once at load time — no need to call inside predict

# ── Transforms: must match training val/test transforms exactly ───────────
# During training, val images were resized and normalized with ImageNet stats.
# Using different transforms here would silently degrade prediction quality.
preprocess = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_img(img_path):
    """
    Load an image from disk and apply the inference transform pipeline.

    Uses PIL directly instead of Keras load_img to remove the TensorFlow
    dependency from the inference path entirely.

    Args:
        img_path (str): Path to the image file.

    Returns:
        torch.Tensor: Shape (1, 3, 224, 224), ready for the model.
    """
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def predict_result(X):
    """
    Run inference and return a human-readable label.

    Args:
        X (torch.Tensor): Preprocessed image tensor, shape (1, 3, 224, 224).

    Returns:
        str: "Dishwasher-safe" or "Not dishwasher-safe"
    """
    with torch.no_grad():
        outputs = model(X)
        pred_idx = outputs.argmax(1).item()
        return LABELS[pred_idx]