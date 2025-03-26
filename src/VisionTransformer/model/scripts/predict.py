import os
import torch as th
import numpy as np
from PIL import Image

import pytorch_lightning as pl

from data.lightning_hdf5 import HDF5DataModule
from model.vit import VisionTransformer
from model.lightning.trainer import TrainerModule
from utils.configuration import Configuration

def sample_closed_loop(cfg: Configuration, checkpoint_path, output_path='samples', num_samples=1, device=0):
    os.makedirs(output_path, exist_ok=True)

    # Set device
    device = th.device(f"cuda:{device}" if th.cuda.is_available() else "cpu")

    # Initialize the model
    if cfg.model.data == 'hdf5':
        model_net = VisionTransformer(cfg=cfg.model)
    else:
        raise NotImplementedError(f"Data {cfg.model.data} not implemented")

    # Load the model checkpoint
    model = TrainerModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        model=model_net,
        strict=False,
        map_location=device
    ).to(device)
    model.eval()

    # Initialize data module and get the test dataset
    data_module = HDF5DataModule(cfg)
    data_module.setup(stage='fit')
    test_loader = data_module.train_dataloader()

    # Sample pairs
    with th.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_samples:
                break

            print(f"Processing sample {idx + 1}/{num_samples}")

            # Get input frames and send to device
            frames = batch[0].to(device)  # Shape: [batch_size, num_frames, H, W]
            batch_size, num_frames, H, W = frames.shape

            for i in range(batch_size):
                # Extract the initial frames for the current sample
                input_frames = frames[i]  # Shape: [num_frames, H, W]

                # Initialize input sequence with cfg.model.num_frames - 1 frames
                num_initial_frames = cfg.model.num_frames - 1
                initial_input_frames = input_frames[:num_initial_frames]  # Shape: [num_initial_frames, H, W]

                # Create a black frame to complete the input sequence
                black_frame = th.zeros_like(initial_input_frames[0])  # Shape: [H, W]
                input_sequence = th.cat([initial_input_frames, black_frame.unsqueeze(0)], dim=0)  # Shape: [num_frames, H, W]

                # Number of prediction steps
                total_prediction_steps = 10  # Adjust as needed

                # List to store all frames (initial + predicted)
                all_frames = [frame.clone().cpu() for frame in initial_input_frames]

                # Closed loop prediction
                for step in range(total_prediction_steps):
                    # Prepare input for the model: add batch dimension
                    model_input = input_sequence.unsqueeze(0)  # Shape: [1, num_frames, H, W]

                    # Run the model 
                    predicted_frame = model(model_input)['output'][0,-1]  # Shape: [H, W]
                    predicted_frame = ((predicted_frame * 255).round() / 255).clamp(0, 1)

                    # Append predicted frame to the list
                    all_frames.append(predicted_frame.clone().cpu())

                    # Prepare the next input sequence
                    # Remove the first frame and append the predicted frame
                    input_sequence = th.cat([input_sequence[1:-1], predicted_frame.unsqueeze(0), black_frame.unsqueeze(0)], dim=0)

                # Save the frames as images
                for frame_idx, frame in enumerate(all_frames):
                    # Convert tensor to numpy array and scale to [0, 255]
                    np_frame = frame.numpy() * 255
                    np_frame = np_frame.astype(np.uint8)

                    # Convert to PIL Image (assuming grayscale)
                    pil_image = Image.fromarray(np_frame.squeeze(), mode='L')

                    # Optionally upscale the image
                    pil_image = pil_image.resize((pil_image.width * 4, pil_image.height * 4), Image.NEAREST)

                    # Save the image
                    pil_image.save(os.path.join(output_path, f'prediction_{idx:05d}_{i:05d}_frame_{frame_idx:05d}.png'))

    print(f"Closed-loop sampling completed. Samples saved to {output_path}")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("-load", "--load", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("-output", "--output", default="samples", type=str, help="Output directory for samples")
    parser.add_argument("-num_samples", "--num_samples", default=1, type=int, help="Number of samples to generate")
    parser.add_argument("-device", "--device", default=0, type=int, help="CUDA device ID")
    parser.add_argument("-seed", "--seed", default=1234, type=int, help="Random seed")

    args = parser.parse_args(sys.argv[1:])
    cfg = Configuration(args.cfg)

    cfg.seed = args.seed
    cfg.model.batch_size = 1
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    sample_closed_loop(
        cfg=cfg,
        checkpoint_path=args.load,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device
    )

