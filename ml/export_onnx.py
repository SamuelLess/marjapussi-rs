"""
Exports the Marjapussi PyTorch model to ONNX format.
This allows the trained model to be loaded directly inside the Rust engine via `ort` (ONNX Runtime)
or `tract`, bypassing Python GIL delays entirely for high-speed simulation clustering.

Usage:
  python export_onnx.py --checkpoint ml/checkpoints/best.pt --output target/model.onnx
"""

import argparse
from pathlib import Path
import torch

from model import ACTION_FEAT_DIM, MarjapussiNet

def export_onnx(checkpoint_path: str, output_path: str):
    device = "cpu"
    model = MarjapussiNet().to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print("Warning: No valid checkpoint provided, exporting randomly initialized model.")
        
    model.eval()

    # Create dummy inputs that match the network signature exactly
    # B = batch dimension (1 for inference)
    # max_s = sequence length (Stream B tokens)
    # max_a = legal actions count (Stream C inputs)
    B, max_s, max_a = 1, 12, 16 

    dummy_tensors = {
        # Stream A: global game state tensors
        "obs_a": {
            "scalar_feats": torch.zeros((B, 12), dtype=torch.float32, device=device),
            "card_status": torch.zeros((B, 36, 12), dtype=torch.float32, device=device),
        },
        # Stream B: token sequence history
        "token_ids": torch.zeros((B, max_s), dtype=torch.long, device=device),
        "token_mask": torch.ones((B, max_s), dtype=torch.bool, device=device),
        
        # Stream C: active legal actions
        "action_feats": torch.zeros((B, max_a, ACTION_FEAT_DIM), dtype=torch.float32, device=device),
        "action_mask": torch.ones((B, max_a), dtype=torch.bool, device=device),
    }

    print(f"Exporting model to {output_path}...")
    
    # We use torch.onnx.export to trace the model's computation graph
    torch.onnx.export(
        model, 
        (dummy_tensors,),                        # model input (must be a tuple)
        output_path,                             # where to save the model
        export_params=True,                      # store the trained parameter weights inside the model file
        opset_version=17,                        # the ONNX version to export the model to
        do_constant_folding=True,                # whether to execute constant folding for optimization
        input_names=["obs_a_scalars", "obs_a_cards", "token_ids", "token_mask", "action_feats", "action_mask"],
        output_names=["logits", "value", "pts_pred", "raw_value"],
        dynamic_axes={
            "obs_a_scalars": {0: "batch_size"}, 
            "obs_a_cards": {0: "batch_size"},
            "token_ids": {0: "batch_size", 1: "seq_length"},
            "token_mask": {0: "batch_size", 1: "seq_length"},
            "action_feats": {0: "batch_size", 1: "num_actions"},
            "action_mask": {0: "batch_size", 1: "num_actions"},
            "logits": {0: "batch_size", 1: "num_actions"},
            "value": {0: "batch_size"},
            "pts_pred": {0: "batch_size"},
            "raw_value": {0: "batch_size"}
        }
    )
    
    print(f"Export successful! The model is ready to be embedded natively into Rust via `ort`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to PyTorch .pt model")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output path for .onnx file")
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output)
