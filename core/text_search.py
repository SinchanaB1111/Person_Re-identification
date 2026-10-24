"""Description-based search: CLIP vision-language embeddings plus simple appearance attributes.

CLIP maps both crops and sentences into one vector space, so "a man in a red jacket"
lands near crops of people in red jackets. On small surveillance crops CLIP alone is
noisy, so we add explicit attributes we can measure directly: the dominant colour of
the upper and lower body, and bags found by the detector. A sentence is parsed into
the same attributes and matching ones add a bonus to the CLIP score.
"""
import re
from collections import Counter

import cv2
import numpy as np
import open_clip
import torch

from .config import CLIP_ARCH, CLIP_WEIGHTS, get_device

COLOURS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]

COLOUR_SYNONYMS = {
    "grey": "gray", "silver": "gray", "navy": "blue", "denim": "blue", "maroon": "red",
    "burgundy": "red", "crimson": "red", "beige": "brown", "khaki": "brown", "tan": "brown",
    "cream": "white", "violet": "purple", "magenta": "pink", "olive": "green",
    "golden": "yellow", "gold": "yellow", "dark": None, "light": None,
}
UPPER_WORDS = {"shirt", "tshirt", "t-shirt", "top", "jacket", "coat", "hoodie", "sweater", "jumper",
               "blouse", "kurta", "vest", "sweatshirt", "cardigan", "blazer", "dress", "tee", "jersey"}
LOWER_WORDS = {"trousers", "pants", "jeans", "shorts", "skirt", "leggings", "joggers", "chinos", "trouser", "pant"}
# Colours of these are left to CLIP (we do not measure shoe or hat colour separately).
OTHER_ITEM_WORDS = {"shoes", "shoe", "sneakers", "trainers", "boots", "sandals", "hat", "cap", "helmet", "beanie",
                    "hair", "scarf", "gloves", "glasses", "sunglasses", "umbrella", "phone"}
BAG_WORDS = {"backpack": "backpack", "rucksack": "backpack", "schoolbag": "backpack",
             "handbag": "handbag", "purse": "handbag", "bag": "bag", "suitcase": "suitcase",
             "luggage": "suitcase", "trolley": "suitcase"}

ATTR_BONUS = 0.04  # added to the CLIP score for each matching attribute

# Colours a camera often confuses (dark navy reads as black, faded denim as gray): half credit.
SIMILAR_COLOURS = {frozenset(p) for p in [("blue", "gray"), ("black", "gray"), ("blue", "black"),
                                          ("brown", "orange"), ("brown", "gray"), ("pink", "red"),
                                          ("purple", "blue"), ("white", "gray")]}


# ---------------------------------------------------------------------------
# Colour attributes measured from a crop
# ---------------------------------------------------------------------------

def _pixel_colour_names(hsv):
    h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    names = np.full(h.shape, "gray", dtype=object)
    chroma = s >= 0.22
    names[chroma & ((h < 8) | (h >= 165))] = "red"
    names[chroma & (h >= 8) & (h < 20)] = "orange"
    names[chroma & (h >= 8) & (h < 25) & (v < 0.6)] = "brown"
    names[chroma & (h >= 20) & (h < 34) & (v >= 0.6)] = "yellow"
    names[chroma & (h >= 34) & (h < 85)] = "green"
    # Denim and navy are only faintly saturated, so blue gets a lower saturation bar.
    names[(s >= 0.12) & (h >= 90) & (h < 130) & (v >= 0.15)] = "blue"
    names[chroma & (h >= 130) & (h < 150)] = "purple"
    names[chroma & (h >= 150) & (h < 165)] = "pink"
    names[~chroma & (v > 0.75)] = "white"
    names[v < 0.22] = "black"
    return names


def _region_colours(crop, top, bottom):
    h, w = crop.shape[:2]
    region = crop[int(h * top):int(h * bottom), int(w * 0.25):int(w * 0.75)]
    if region.size == 0:
        return []
    region = cv2.resize(region, (24, 24), interpolation=cv2.INTER_AREA)
    names = _pixel_colour_names(cv2.cvtColor(region, cv2.COLOR_BGR2HSV)).ravel()
    return [c for c, _ in Counter(names).most_common(2)]


def crop_colours(crop_bgr):
    """Return (upper_colours, lower_colours), each the top-2 colour names of that body region."""
    return _region_colours(crop_bgr, 0.18, 0.50), _region_colours(crop_bgr, 0.55, 0.85)


# ---------------------------------------------------------------------------
# Parsing a description into attributes
# ---------------------------------------------------------------------------

def _colour_of(word):
    word = word.lower()
    if word in COLOURS:
        return word
    return COLOUR_SYNONYMS.get(word)


def parse_description(text):
    """'man in a blue shirt and black jeans with a backpack' ->
    {'upper': 'blue', 'lower': 'black', 'any': [], 'bags': ['backpack']}"""
    tokens = re.findall(r"[a-zA-Z\-]+", text.lower())
    attrs = {"upper": None, "lower": None, "any": [], "bags": []}
    used = set()
    for i, tok in enumerate(tokens):
        region = "upper" if tok in UPPER_WORDS else "lower" if tok in LOWER_WORDS else None
        if region:
            for j in range(i - 1, max(-1, i - 4), -1):
                colour = _colour_of(tokens[j])
                if colour:
                    attrs[region] = colour
                    used.add(j)
                    break
        if tok in BAG_WORDS and not (i > 0 and tokens[i - 1] in {"no", "without"}):
            attrs["bags"].append(BAG_WORDS[tok])
    for i, tok in enumerate(tokens):
        colour = _colour_of(tok)
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if colour and i not in used and nxt not in BAG_WORDS and nxt not in OTHER_ITEM_WORDS:
            attrs["any"].append(colour)
    return attrs


def attribute_bonus(attrs, upper, lower, bags):
    """Score how well a track's measured attributes agree with the parsed description.

    `upper` / `lower` are the track's top-2 colours (first is dominant); `bags` is a set.
    Returns (bonus, list of human-readable matched attributes).
    """
    bonus, matched = 0.0, []

    def colour_score(wanted, have):
        if not wanted or not have:
            return 0.0
        if have[0] == wanted:
            return 1.0
        if wanted in have[1:] or frozenset((wanted, have[0])) in SIMILAR_COLOURS:
            return 0.5
        return 0.0

    for region, have in (("upper", upper), ("lower", lower)):
        s = colour_score(attrs[region], have)
        if s:
            bonus += ATTR_BONUS * s
            matched.append(f"{attrs[region]} {region}")
    for colour in attrs["any"]:
        if colour in (upper or [])[:1] + (lower or [])[:1]:
            bonus += ATTR_BONUS * 0.5
            matched.append(colour)
    for bag in attrs["bags"]:
        if (bag == "bag" and bags) or bag in bags:
            bonus += ATTR_BONUS
            matched.append(bag)
    return bonus, matched


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

class ClipModel:
    def __init__(self, weights=CLIP_WEIGHTS, device=None):
        self.device = device or get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_ARCH, pretrained=str(weights), device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_ARCH)

    @torch.no_grad()
    def encode_images(self, crops_bgr, batch_size=32):
        from PIL import Image
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        out = []
        for i in range(0, len(crops_bgr), batch_size):
            imgs = [self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                    for c in crops_bgr[i:i + batch_size]]
            f = self.model.encode_image(torch.stack(imgs).to(self.device))
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text):
        """One description -> one 512-d vector (averaged over a few prompt templates)."""
        text = text.strip().rstrip(".")
        prompts = [f"a photo of {text}", f"a surveillance camera photo of {text}",
                   f"a full-body photo of a person: {text}"]
        f = self.model.encode_text(self.tokenizer(prompts).to(self.device))
        f = torch.nn.functional.normalize(f, dim=1).mean(0)
        return torch.nn.functional.normalize(f, dim=0).cpu().numpy().astype(np.float32)
