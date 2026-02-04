import os, glob, argparse, cv2, numpy as np, torch
from models.stitchnet import StitchNet


def infer(folder, weights, out_name='stitched.png', size=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = StitchNet().to(device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()

    files = sorted(glob.glob(os.path.join(folder, '*.png')))                              # read tiles [3]
    C = int(len(files) ** 0.5)
    R = int(np.ceil(len(files) / C))
    canvas = np.zeros((R * size, C * size, 3), np.uint8)

    for f in files:
        idx = int(os.path.splitext(os.path.basename(f))[0].split('_')[-1])
        r, c = divmod(idx, C)
        img = cv2.resize(cv2.imread(f), (size, size))
        t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t = t.to(device)
        warped, _ = net(t, t)                                                             # identity neighbour
        out = (warped.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        canvas[r * size:(r + 1) * size, c * size:(c + 1) * size] = out
    cv2.imwrite(out_name, canvas)
    print(f'Saved {out_name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True, help='folder with PNG tiles')
    ap.add_argument('--weights', default='models/stitchnet_ep10.pth')
    args = ap.parse_args()
    infer(args.folder, args.weights)


if __name__ == '__main__':
    main()
