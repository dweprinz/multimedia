#!/usr/bin/env python3
"""
Generate Embeddings Script

This script generates embeddings for an existing hierarchical tree structure (created by create_tree_structure.py).
It loads a model (e.g., HyCoCLIP) and processes each node (image/text) in the tree, saving the embeddings to file.

Outputs:
- embeddings.pkl or embeddings.json (embeddings for all nodes)

Usage:
    python generate_embeddings.py --dataset <DATASET_NAME> --checkpoint_path <MODEL_CHECKPOINT> --train_config <CONFIG_PATH> [other args]
"""

import os
import json
import pickle
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from torchvision import transforms as T
from datetime import datetime

# HyCoCLIP imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip'))
from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.hycoclip_utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_hycoclip_model(checkpoint_path, train_config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint_path or not train_config_path:
        raise ValueError("Both checkpoint_path and train_config_path are required for HyCoCLIP model")
    train_config = LazyConfig.load(train_config_path)
    model = LazyFactory.build_model(train_config, device).eval()
    CheckpointManager(model=model).load(checkpoint_path)
    preprocess = T.Compose([
        T.Resize(224, T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    return model, preprocess, device

def generate_text_embedding(model, text, device):
    tokenizer = Tokenizer()
    text_tokens = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens, project=True)
    return text_features.cpu().numpy().flatten()

def generate_image_embedding(model, preprocess, image_path, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor, project=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for an existing tree structure.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ImageNet)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--train_config', type=str, required=True, help='Path to model config (for HyCoCLIP)')
    parser.add_argument('--output_dir', type=str, default='hierchical_datasets', help='Output directory')
    args = parser.parse_args()

    dataset_path = Path(args.output_dir) / args.dataset
    meta_path = dataset_path / "meta_data_trees.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta_data_trees.json not found in {dataset_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    model, preprocess, device = load_hycoclip_model(args.checkpoint_path, args.train_config)

    embeddings = {}
    for tree_id, tree in tqdm(meta_data["trees"].items(), desc="Generating embeddings for trees"):
        # Text embeddings
        parent_text = tree["parent_text"]["text"]
        child_text = tree["child_text"]["text"]
        parent_text_id = f"pt_{tree_id}"
        child_text_id = f"ct_{tree_id}"
        embeddings[parent_text_id] = generate_text_embedding(model, parent_text, device)
        embeddings[child_text_id] = generate_text_embedding(model, child_text, device)
        # Image embeddings
        for i, img_path in enumerate(tree["child_images"]):
            img_id = f"ci_{tree_id}_{i+1:03d}"
            emb = generate_image_embedding(model, preprocess, img_path, device)
            if emb is not None:
                embeddings[img_id] = emb
        # (Add parent_images if needed)
    # Save embeddings
    emb_out = dataset_path / "embeddings.pkl"
    with open(emb_out, 'wb') as f:
        pickle.dump({"embeddings": embeddings, "created_at": datetime.now().isoformat()}, f)
    print(f"Saved embeddings to {emb_out}")

if __name__ == '__main__':
    main() 