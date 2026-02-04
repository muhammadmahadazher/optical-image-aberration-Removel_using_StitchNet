import os, glob, random, argparse, torch, torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.stitchnet import StitchNet
from models.losses  import total_loss
import torchvision.io                                   # keep the fast C-backend

class PandaTiles(Dataset):
    def __init__(self, root, crop=3, size=512):
        self.root, self.crop, self.size = root, crop, size
        self.resize = lambda t: F.interpolate(t.unsqueeze(0), size=(size, size),
                                              mode='bilinear', align_corners=False
                                              ).squeeze(0)
        # ---- detect layout (flat vs. slide sub-dirs) ----
        flat = [f for f in os.listdir(root) if f.lower().endswith('.png')]
        if flat:                                                # flat folder
            from collections import defaultdict
            by = defaultdict(list)
            for f in flat:
                sid = f.rsplit('_', 1)[0]
                by[sid].append(f)
            self.slides, self.by_slide, self.flat = list(by.keys()), by, True
        else:                                                   # hierarchical
            self.slides, self.flat = [d for d in os.listdir(root)
                                       if os.path.isdir(os.path.join(root, d))], False

    def __len__(self): return len(self.slides)

    def _tile_list(self, slide):
        if self.flat:
            paths = [os.path.join(self.root, f) for f in self.by_slide[slide]]
        else:
            paths = glob.glob(os.path.join(self.root, slide, '*.png'))
        return sorted(paths, key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))

    def __getitem__(self, idx):
        paths  = self._tile_list(self.slides[idx])
        anchor = random.randint(0, len(paths) - self.crop ** 2)
        imgs   = []
        for k in range(self.crop ** 2):
            t = torchvision.io.read_image(paths[anchor + k]).float() / 255.0  # already tensor[11]
            if t.shape[-1] != self.size: t = self.resize(t)
            imgs.append(t)
        ref, nbr, target = imgs[0], imgs[1], imgs[-1]
        return ref, nbr, target

def train(dataset_dir, epochs=10, batch=4, lr=3e-4, workers=4):
    ds = PandaTiles(dataset_dir)
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    num_workers=workers, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = StitchNet().to(device)  # create the model first
    opt = torch.optim.AdamW(net.parameters(), lr)  # now net exists

    net, opt = StitchNet().to(device), torch.optim.AdamW(net.parameters(), lr)
    for ep in range(1, epochs + 1):
        net.train(); running = 0.0
        for ref, nbr, tgt in tqdm(dl, desc=f'Epoch {ep}/{epochs}'):
            ref, nbr, tgt = [x.to(device) for x in (ref, nbr, tgt)]
            warp, flow = net(ref, nbr)
            loss = total_loss(warp, tgt, flow)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * ref.size(0)
        print(f'Epoch {ep}: mean loss {running / len(ds):.4f}')
        torch.save(net.state_dict(), f'models/stitchnet_ep{ep}.pth')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles_dir', default='data/panda_tiles')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch',  type=int, default=4)
    ap.add_argument('--workers',type=int, default=4,
                    help='set 0 on Windows for easier debugging')
    args = ap.parse_args()
    train(args.tiles_dir, args.epochs, args.batch, workers=args.workers)
