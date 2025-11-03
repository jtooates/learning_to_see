"""Data generation pipeline: scene graphs -> images + text."""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

from dsl.parser import SceneParser
from dsl.canonicalize import to_canonical
from dsl.splits import enumerate_all_samples, make_split_indices
from render.renderer import SceneRenderer


def sample_scene_graphs(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample n scene graphs with balanced distribution.

    Args:
        n: Number of samples to generate
        seed: Random seed

    Returns:
        List of scene graph dictionaries
    """
    rng = random.Random(seed)

    # Get all possible scene graphs
    all_graphs = enumerate_all_samples()
    print(f"Total possible scenes: {len(all_graphs)}")

    # Sample with replacement if n > total
    if n <= len(all_graphs):
        sampled = rng.sample(all_graphs, n)
    else:
        # Sample with replacement
        sampled = [rng.choice(all_graphs) for _ in range(n)]

    return sampled


def generate_shard(scene_graphs: List[Dict[str, Any]],
                   shard_idx: int,
                   out_dir: Path,
                   renderer: SceneRenderer,
                   parser: SceneParser) -> Dict[str, int]:
    """Generate a single data shard.

    Args:
        scene_graphs: List of scene graphs to render
        shard_idx: Shard index
        out_dir: Output directory
        renderer: SceneRenderer instance
        parser: SceneParser instance

    Returns:
        Statistics dictionary
    """
    images = []
    texts = []
    graphs = []
    metadata_list = []

    stats = {"success": 0, "failed": 0}

    for graph in tqdm(scene_graphs, desc=f"Shard {shard_idx}"):
        try:
            # Canonicalize to text
            canonical_text = to_canonical(graph)

            # Render image
            image, meta = renderer.render(graph)

            # Convert image to tensor (C, H, W)
            img_array = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
            img_tensor = torch.from_numpy(img_array).float() / 255.0

            # Store
            images.append(img_tensor)
            texts.append(canonical_text)
            graphs.append(graph)

            # Store metadata (centers, bboxes, masks)
            meta_dict = {
                "instances": [
                    {
                        "object_id": inst.object_id,
                        "shape": inst.shape,
                        "color": inst.color,
                        "center": inst.center,
                        "bbox": inst.bbox,
                        "mask": inst.mask
                    }
                    for inst in meta.instances
                ],
                "relations_valid": meta.relations_valid
            }
            metadata_list.append(meta_dict)

            stats["success"] += 1

        except Exception as e:
            print(f"Failed to render graph: {e}")
            stats["failed"] += 1
            continue

    # Save shard
    if images:
        # Stack images into tensor (N, C, H, W)
        images_tensor = torch.stack(images)

        # Save images
        torch.save(images_tensor, out_dir / f"images_{shard_idx:04d}.pt")

        # Save texts (JSONL)
        with open(out_dir / f"texts_{shard_idx:04d}.jsonl", 'w') as f:
            for text in texts:
                f.write(json.dumps({"text": text}) + '\n')

        # Save graphs (JSONL)
        with open(out_dir / f"graphs_{shard_idx:04d}.jsonl", 'w') as f:
            for graph in graphs:
                f.write(json.dumps(graph) + '\n')

        # Save metadata
        torch.save(metadata_list, out_dir / f"meta_{shard_idx:04d}.pt")

        print(f"Saved shard {shard_idx}: {len(images)} samples")

    return stats


def parse_split_strategy(strategy_str: str) -> Tuple[str, Dict[str, Any]]:
    """Parse split strategy from command line string.

    Args:
        strategy_str: Strategy string like "color_shape:yellow,cube"

    Returns:
        Tuple of (strategy_name, kwargs)
    """
    if ':' in strategy_str:
        strategy, args = strategy_str.split(':', 1)
    else:
        strategy = strategy_str
        args = ''

    kwargs = {}

    if strategy == 'color_shape' and args:
        # Parse "yellow,cube" or "yellow,cube;red,ball"
        pairs = []
        for pair_str in args.split(';'):
            color, shape = pair_str.split(',')
            pairs.append((color.strip(), shape.strip()))
        kwargs['holdout_pairs'] = pairs

    elif strategy == 'count_shape' and args:
        # Parse "5,ball" or "5,ball;4,cube"
        pairs = []
        for pair_str in args.split(';'):
            count_str, shape = pair_str.split(',')
            pairs.append((int(count_str.strip()), shape.strip()))
        kwargs['holdout_pairs'] = pairs

    elif strategy == 'relation' and args:
        # Parse "in_front_of" or "in_front_of;on"
        relations = [rel.strip() for rel in args.split(';')]
        kwargs['holdout_relations'] = relations

    return strategy, kwargs


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic scene dataset')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for generated data')
    parser.add_argument('--n', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--shard_size', type=int, default=1000,
                       help='Number of samples per shard')
    parser.add_argument('--split_strategy', type=str, default='random',
                       help='Split strategy (random, color_shape:yellow,cube, count_shape:5,ball, relation:in_front_of)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size (square)')

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} samples to {out_dir}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Seed: {args.seed}")

    # Initialize components
    scene_parser = SceneParser()
    renderer = SceneRenderer(
        width=args.image_size,
        height=args.image_size,
        seed=args.seed
    )

    # Sample scene graphs
    print("\nSampling scene graphs...")
    scene_graphs = sample_scene_graphs(args.n, seed=args.seed)

    # Create splits
    print("\nCreating data splits...")
    strategy_name, strategy_kwargs = parse_split_strategy(args.split_strategy)
    splits = make_split_indices(
        scene_graphs,
        strategy=strategy_name,
        seed=args.seed,
        **strategy_kwargs
    )

    # Save split indices
    split_info = {
        'strategy': args.split_strategy,
        'train_indices': splits['train'],
        'val_indices': splits['val'],
        'test_indices': splits['test'],
        'total': args.n
    }
    with open(out_dir / 'splits.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # Generate data in shards
    print("\nGenerating data shards...")
    total_stats = {"success": 0, "failed": 0}

    n_shards = (args.n + args.shard_size - 1) // args.shard_size

    for shard_idx in range(n_shards):
        start_idx = shard_idx * args.shard_size
        end_idx = min(start_idx + args.shard_size, args.n)

        shard_graphs = scene_graphs[start_idx:end_idx]

        # Reinitialize renderer with different seed per shard for variety
        renderer = SceneRenderer(
            width=args.image_size,
            height=args.image_size,
            seed=args.seed + shard_idx
        )

        stats = generate_shard(
            shard_graphs,
            shard_idx,
            out_dir,
            renderer,
            scene_parser
        )

        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]

    # Save manifest
    manifest = {
        'n_samples': args.n,
        'n_shards': n_shards,
        'shard_size': args.shard_size,
        'image_size': args.image_size,
        'seed': args.seed,
        'split_strategy': args.split_strategy,
        'stats': total_stats
    }

    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGeneration complete!")
    print(f"Success: {total_stats['success']}, Failed: {total_stats['failed']}")
    print(f"Data saved to: {out_dir}")


if __name__ == '__main__':
    main()
