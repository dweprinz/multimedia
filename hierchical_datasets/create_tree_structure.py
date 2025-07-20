#!/usr/bin/env python3
"""
Create Tree Structure Script

This script processes raw datasets (e.g., ImageNet, GRIT) to create a hierarchical tree structure and metadata files, but does NOT generate embeddings.

Outputs:
- tree folders (tree1, tree2, ...)
- meta_data_trees.json (tree structure and metadata)

Usage:
    python create_tree_structure.py --dataset <DATASET_NAME> [other args]
"""

import os
import csv
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

# Utility functions for reading synsets and creating tree structure

def read_synsets_csv(csv_path):
    synsets = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            synsets.append({
                'synset_id': row['synset_id'],
                'synset_name': row['synset_name'],
                'definition': row['definition']
            })
    return synsets


def create_imagenet_tree_structure(base_path, output_dir, synsets):
    imagenet_path = Path(output_dir) / "ImageNet"
    imagenet_path.mkdir(exist_ok=True)
    data_path = imagenet_path / "data"
    data_path.mkdir(exist_ok=True)
    print(f"Processing {len(synsets)} ImageNet synsets...")
    meta_data = {
        "dataset_info": {
            "name": "ImageNet",
            "total_trees": len(synsets),
            "created_at": datetime.now().isoformat(),
        },
        "trees": {}
    }
    for i, synset in enumerate(tqdm(synsets, desc="Creating ImageNet tree structure")):
        tree_id = f"tree{i+1}"
        tree_folder = data_path / tree_id
        tree_folder.mkdir(exist_ok=True)
        child_images_folder = tree_folder / "child_images"
        parent_images_folder = tree_folder / "parent_images"
        child_images_folder.mkdir(exist_ok=True)
        parent_images_folder.mkdir(exist_ok=True)
        parent_texts_folder = tree_folder / "parent_texts"
        child_texts_folder = tree_folder / "child_texts"
        parent_texts_folder.mkdir(exist_ok=True)
        child_texts_folder.mkdir(exist_ok=True)
        # Save raw text data to txt files for easy inspection
        parent_text_file = parent_texts_folder / "parent_text.txt"
        child_text_file = child_texts_folder / "child_text.txt"
        with open(parent_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Synset ID: {synset['synset_id']}\n")
            f.write(f"Synset Name: {synset['synset_name']}\n")
            f.write(f"Tree: {tree_id}\n")
        with open(child_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Synset ID: {synset['synset_id']}\n")
            f.write(f"Definition: {synset['definition']}\n")
            f.write(f"Tree: {tree_id}\n")
        # Copy images from the original synset folder
        original_folder = Path(base_path) / synset['synset_id']
        if original_folder.exists():
            image_files = list(original_folder.glob("*.JPEG"))
            for j, image_path in enumerate(image_files):
                safe_synset_name = synset['synset_name'].replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                new_image_name = f"{safe_synset_name}_{j+1:03d}.JPEG"
                new_image_path = child_images_folder / new_image_name
                shutil.copy(image_path, new_image_path)
        # Add tree metadata
        meta_data["trees"][tree_id] = {
            "synset_id": synset['synset_id'],
            "synset_name": synset['synset_name'],
            "definition": synset['definition'],
            "tree_id": tree_id,
            "child_images": [str(p) for p in child_images_folder.glob("*.JPEG")],
            "parent_images": [],
            "parent_text": {"text": synset['synset_name']},
            "child_text": {"text": synset['definition']},
        }
    # Save meta_data_trees.json
    with open(imagenet_path / "meta_data_trees.json", 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2)
    print(f"Saved tree structure and metadata to {imagenet_path / 'meta_data_trees.json'}")

# Main CLI

def main():
    parser = argparse.ArgumentParser(description="Create tree structure and metadata for a dataset (no embeddings).")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., ImageNet)')
    parser.add_argument('--synsets_csv', type=str, help='Path to synsets.csv (for ImageNet)')
    parser.add_argument('--base_path', type=str, help='Path to raw images (for ImageNet)')
    parser.add_argument('--output_dir', type=str, default='hierchical_datasets', help='Output directory')
    args = parser.parse_args()

    if args.dataset.lower() == 'imagenet':
        if not args.synsets_csv or not args.base_path:
            raise ValueError('For ImageNet, --synsets_csv and --base_path are required')
        synsets = read_synsets_csv(args.synsets_csv)
        create_imagenet_tree_structure(args.base_path, args.output_dir, synsets)
    else:
        raise NotImplementedError('Only ImageNet tree creation is implemented in this script. Add GRIT or other dataset support as needed.')

if __name__ == '__main__':
    main() 