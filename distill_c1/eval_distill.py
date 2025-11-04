"""
Evaluation script for renderer distillation.

CLI:
  python -m distill_c1.eval_distill \\
    --data_dir data/scenes \\
    --ckpt runs/distill_c1/ema_best.pt \\
    --report runs/distill_c1/report.json
"""

import argparse
import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm

from .text_encoder import build_text_encoder
from .decoder import build_decoder
from .trainer import DistillDataset
from .metrics import compute_psnr, compute_ssim, counterfactual_sensitivity, MetricsTracker
from .vis import save_grid, save_comparison
from dsl.tokens import Vocab
from dsl.parser import SceneParser
from render.renderer import SceneRenderer


def main():
    parser = argparse.ArgumentParser(description='Evaluate text-to-image distillation model')

    # Required
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing scene data')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint')

    # Optional
    parser.add_argument('--report', type=str, default=None,
                        help='Path to save evaluation report (JSON)')
    parser.add_argument('--save_images', type=str, default=None,
                        help='Directory to save output images')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--split', type=str, default='val',
                        help='Split to evaluate (val or test)')
    parser.add_argument('--counterfactual', action='store_true',
                        help='Perform counterfactual sensitivity analysis')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')

    args = parser.parse_args()

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("Renderer Distillation Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = Vocab()

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=args.device)

    # Build models
    text_encoder = build_text_encoder(vocab_size=len(vocab), pad_id=vocab.pad_id)
    decoder = build_decoder()

    # Load weights
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    text_encoder = text_encoder.to(args.device)
    decoder = decoder.to(args.device)

    text_encoder.eval()
    decoder.eval()

    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = DistillDataset(
        data_dir=args.data_dir,
        split=args.split,
        vocab=vocab,
    )

    # Limit samples if specified
    if args.max_samples:
        dataset.dataset.indices = dataset.dataset.indices[:args.max_samples]

    print(f"Evaluating on {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
    )

    # Evaluate
    print("\nEvaluating...")
    metrics = MetricsTracker()

    # Collect images for visualization
    all_teacher = []
    all_student = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            token_ids = batch['token_ids'].to(args.device)
            target_images = batch['image'].to(args.device)
            texts = batch['text']

            # Forward pass
            e = text_encoder(token_ids, pad_id=vocab.pad_id)
            pred_images = decoder(e)

            # Compute metrics
            psnr = compute_psnr(pred_images, target_images)
            ssim = compute_ssim(pred_images, target_images)

            metrics.update(psnr=psnr, ssim=ssim)

            # Collect for visualization
            all_teacher.append(target_images.cpu())
            all_student.append(pred_images.cpu())
            all_texts.extend(texts)

    # Get average metrics
    avg_metrics = metrics.get_averages()

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print("=" * 60)

    # Counterfactual sensitivity analysis
    counterfactual_results = {}
    if args.counterfactual:
        print("\nPerforming counterfactual sensitivity analysis...")

        scene_parser = SceneParser()
        renderer = SceneRenderer()

        # Sample a few scenes for counterfactual analysis
        test_texts = [
            "There is one red ball.",
            "There are two green cubes.",
            "The blue block is left of the yellow ball.",
            "The red cube is on the green block.",
        ]

        for edit_type in ['color', 'shape', 'number', 'relation']:
            print(f"\nTesting {edit_type} edits...")
            edit_results = []

            for text in test_texts:
                try:
                    result = counterfactual_sensitivity(
                        text_encoder=text_encoder,
                        decoder=decoder,
                        vocab=vocab,
                        parser=scene_parser,
                        renderer=renderer,
                        original_text=text,
                        token_edit=edit_type,
                        device=args.device,
                    )
                    edit_results.append({
                        'original_text': result['original_text'],
                        'edited_text': result['edited_text'],
                        'delta_l2': result['delta_l2'],
                        'delta_psnr': result['delta_psnr'],
                    })
                    print(f"  {text} → {result['edited_text']}")
                    print(f"    ΔL2: {result['delta_l2']:.6f}")
                except Exception as e:
                    print(f"  Skipping {text}: {e}")

            if edit_results:
                avg_delta_l2 = sum(r['delta_l2'] for r in edit_results) / len(edit_results)
                counterfactual_results[edit_type] = {
                    'results': edit_results,
                    'avg_delta_l2': avg_delta_l2,
                }
                print(f"  Average ΔL2: {avg_delta_l2:.6f}")

    # Save report
    report = {
        'checkpoint': args.ckpt,
        'step': checkpoint.get('step', None),
        'split': args.split,
        'num_samples': len(dataset),
        'metrics': avg_metrics,
        'counterfactual': counterfactual_results,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {args.report}")

    # Save visualizations
    if args.save_images:
        save_dir = Path(args.save_images)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving visualizations to {save_dir}...")

        # Concatenate all images
        all_teacher = torch.cat(all_teacher[:4], dim=0)  # First 4 batches
        all_student = torch.cat(all_student[:4], dim=0)

        # Save grid
        save_grid(
            teacher=all_teacher[:32],
            student=all_student[:32],
            path=str(save_dir / 'grid.png'),
            nrow=8,
        )

        # Save comparison
        save_comparison(
            teacher=all_teacher[:16],
            student=all_student[:16],
            path=str(save_dir / 'comparison.png'),
            max_images=16,
        )

        print(f"✓ Visualizations saved")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
