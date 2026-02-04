import os, shutil, argparse, re
p = argparse.ArgumentParser()
p.add_argument("--in",  dest="src", required=True)
p.add_argument("--out", dest="dst", default=None)
args = p.parse_args()

src, dst = args.src, args.dst or args.src          # in-place by default
pat  = re.compile(r"([a-f0-9]{32})_(\d+)\.png$",re.I)

for f in os.listdir(src):
    m = pat.match(f)
    if not m:
        print("Skip", f); continue
    slide = m.group(1)
    tgt_dir = os.path.join(dst, slide)
    os.makedirs(tgt_dir, exist_ok=True)
    shutil.move(os.path.join(src,f), os.path.join(tgt_dir,f))
print("Completed!")
