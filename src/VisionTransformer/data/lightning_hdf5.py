import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from utils.configuration import Configuration
import h5py
from PIL import Image
import io

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, train=True, num_frames=3):
        self.hdf5_file_path = hdf5_file
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        self.train = train
        self.num_frames = num_frames

        # Read the number of images and create indices for consecutive frames
        with h5py.File(self.hdf5_file_path, 'r') as f:
            self.num_images = f['/rgb_images'].shape[0]
        self.indices = list(range(self.num_images - self.num_frames + 1))

        # Placeholder for the HDF5 file, will be opened per worker
        self.hdf5_file = None
        print(f"Dataset created with {len(self)} samples in mode {'train' if train else 'test'}")

    def __len__(self):
        # Updated to reflect the number of 3-frame sequences available
        return len(self.indices)*100 if self.train else len(self.indices)

    def __getitem__(self, idx):
        # Get the starting index of the 3-frame sequence
        index = self.indices[idx%len(self.indices)]

        # Open the HDF5 file per worker if not already opened
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, 'r', swmr=True)
            self.rgb_images = self.hdf5_file['/rgb_images']

        # Load and transform three consecutive frames
        frames = []
        for i in range(self.num_frames):
            img_bytes = self.rgb_images[index + i]
            image = Image.open(io.BytesIO(img_bytes)).convert("L")
            frames.append(self.transform(image))

        # concatenate the frames along the channel dimension
        frames = th.cat(frames, dim=0)  # Shape: [ N, H, W]

        # Return the stacked frames and a placeholder label (e.g., 0)
        return frames, 0

    def __del__(self):
        # Ensure that the HDF5 file is closed when the dataset is deleted
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None


class HDF5DataModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.num_workers = cfg.num_workers
        self.batch_size  = cfg.model.batch_size
        self.hdf5_file   = cfg.data.hdf5_file  # Assuming you have this in your configuration
        self.num_frames  = cfg.model.num_frames

    def setup(self, stage=None):
        # Initialize the datasets
        if stage == 'fit' or stage is None:
            self.hdf5_train = HDF5Dataset(hdf5_file=self.hdf5_file, train=True, num_frames=self.num_frames)
            self.hdf5_val = HDF5Dataset(hdf5_file=self.hdf5_file, train=False, num_frames=self.num_frames)
        
        if stage == 'test' or stage is None:
            self.hdf5_test = HDF5Dataset(hdf5_file=self.hdf5_file, train=False, num_frames=self.num_frames)

    def train_dataloader(self):
        return DataLoader(self.hdf5_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.hdf5_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.hdf5_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

