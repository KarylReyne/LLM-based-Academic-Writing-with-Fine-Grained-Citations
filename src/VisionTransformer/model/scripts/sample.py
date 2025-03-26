import os
import torch as th
import numpy as np
from PIL import Image

import pytorch_lightning as pl

from data.lightning_hdf5 import HDF5DataModule
from model.vit import VisionTransformer
from model.lightning.trainer import TrainerModule
from utils.configuration import Configuration
from tqdm import tqdm

def sample(cfg: Configuration, checkpoint_path, output_path='samples', num_samples=1, device=0):
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
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    num_samples = min(num_samples, len(test_loader))
    pb = tqdm(total=num_samples, desc="Generating samples")

    # Sample pairs
    with th.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_samples:
                pb.close()
                break

            pb.update(1)

            # Get input and send to device
            input_image = batch[0].to(device)
            
            # Get reconstructed image
            output_image = model(input_image)['output'].clamp(0, 1)

            # Process and save input-output pairs
            for i in range(input_image.size(0)):
                inp_img = input_image[i].cpu()
                out_img = output_image[i].cpu()

                # Convert to numpy and scale to [0, 255]
                np_input_img = inp_img.numpy() * 255
                np_output_img = out_img.numpy() * 255

                np_input_img = np_input_img.astype(np.uint8)
                np_output_img = np_output_img.astype(np.uint8)

                # save each channel as a separate image
                combined_img = []
                for c in range(np_input_img.shape[0]):

                    # Concatenate input and output images side-by-side
                    combined_img.append(np.concatenate((np_input_img[c], np_output_img[c]), axis=1))

                combined_img = np.concatenate(combined_img, axis=0)

                # Convert to PIL greayscale and save
                pil_image = Image.fromarray(combined_img)

                # upscale the image
                pil_image = pil_image.resize((pil_image.width * 4, pil_image.height * 4), Image.NEAREST)

                pil_image.save(os.path.join(output_path, f'sample_{idx:05d}_{i:05d}.png'))

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

    sample(
        cfg=cfg,
        checkpoint_path=args.load,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device
    )
