# train_stitch_v2.py
import os
import glob
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as T
from collections import defaultdict
from PIL import Image  # CRITICAL FIX: Import the Image class

# Import your existing, unchanged model and loss files
from models.stitchnet import StitchNet
from models.losses import total_loss


class PandaNeighborDataset(Dataset):
    """
    A PyTorch Dataset that loads true adjacent (horizontal and vertical)
    tile pairs from the PANDA dataset for training a stitching model.
    """

    def __init__(self, root_dir, tile_size=512):
        self.root_dir = root_dir
        self.tile_size = tile_size
        self.transforms = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize([tile_size, tile_size], antialias=True),
        ])

        self.tile_pairs = self._create_neighbor_pairs()

    def _create_neighbor_pairs(self):
        """
        Scans the directory to find all valid horizontal and vertical
        neighboring tiles and returns them as a list of path pairs.
        """
        print("🔍 Scanning for adjacent tile pairs...")
        slide_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        all_pairs = []

        for slide in tqdm(slide_folders, desc="🔬 Processing slides"):
            slide_path = os.path.join(self.root_dir, slide)
            paths = sorted(glob.glob(os.path.join(slide_path, '*.png')),
                           key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]))

            if not paths:
                continue

            # Infer grid width to find vertical neighbors
            indices = [int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in paths]
            path_dict = {idx: p for idx, p in zip(indices, paths)}

            # Simple heuristic for column count
            run, best_run = 1, 1
            for i in range(1, len(indices)):
                if indices[i] == indices[i - 1] + 1:
                    run += 1
                else:
                    best_run = max(best_run, run)
                    run = 1
            cols = max(best_run, run)

            for idx, ref_path in zip(indices, paths):
                # Horizontal neighbor
                if (idx + 1) in path_dict:
                    all_pairs.append((ref_path, path_dict[idx + 1]))
                # Vertical neighbor
                if (idx + cols) in path_dict:
                    all_pairs.append((ref_path, path_dict[idx + cols]))

        print(f"✅ Found {len(all_pairs)} neighboring tile pairs.")
        return all_pairs

    def __len__(self):
        return len(self.tile_pairs)

    def __getitem__(self, idx):
        ref_path, nbr_path = self.tile_pairs[idx]

        ref_img = self.transforms(Image.open(ref_path).convert("RGB"))
        nbr_img = self.transforms(Image.open(nbr_path).convert("RGB"))

        return ref_img, nbr_img


def train(data_dir, epochs, batch_size, learning_rate, workers):
    """
    The main training loop.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    # Initialize Dataset and DataLoader
    dataset = PandaNeighborDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Initialize Model and Optimizer
    model = StitchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("🧠 Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"📅 Epoch {epoch}/{epochs}")
        for ref_tiles, nbr_tiles in progress_bar:
            ref_tiles = ref_tiles.to(device)
            nbr_tiles = nbr_tiles.to(device)

            # Forward pass
            warped_tiles, flow = model(ref_tiles, nbr_tiles)

            # The reference tile itself is the target for the warped neighbor
            loss = total_loss(warped_tiles, ref_tiles, flow)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / len(dataloader)
        print(f"✨ Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

        # Save a checkpoint after each epoch
        torch.save(model.state_dict(), f"models/stitchnet_v2_ep{epoch}.pth")
        print(f"💾 Saved checkpoint to models/stitchnet_v2_ep{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a StitchNet model on the PANDA dataset.")
    parser.add_argument("--tiles_dir", type=str, default="data/panda_tiles",
                        help="Directory containing the PANDA tile folders.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of DataLoader workers. Set to 0 for debugging on Windows.")

    args = parser.parse_args()

    train(data_dir=args.tiles_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
          workers=args.workers)
