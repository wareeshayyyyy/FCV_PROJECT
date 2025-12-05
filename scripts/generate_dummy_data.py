"""Generate a tiny dummy dataset (PPM images + CSV) without external libs.

Creates 3 random 224x224 PPM images at `data/images/` and a `data/labels.csv`.
"""
import os
import random

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
out_dir = os.path.abspath(out_dir)
images_dir = os.path.join(out_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

width, height = 224, 224
n = 3
labels = []
for i in range(n):
    fname = f'img_{i}.ppm'
    path = os.path.join(images_dir, fname)
    # P6 header then random bytes
    with open(path, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode('ascii'))
        # write random RGB bytes
        img_bytes = bytearray(random.getrandbits(8) for _ in range(width * height * 3))
        f.write(img_bytes)
    labels.append((os.path.join('images', fname).replace('\\', '/'), i % 3))

# write CSV
csv_path = os.path.join(out_dir, 'labels.csv')
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write('image_path,label\n')
    for p, l in labels:
        f.write(f'{p},{l}\n')

print(f'Wrote {n} images to {images_dir}')
print(f'Wrote CSV to {csv_path}')
