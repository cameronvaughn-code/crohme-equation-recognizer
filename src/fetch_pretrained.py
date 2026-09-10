"""
fetch_pretrained.py

Downloads the ImageNet-pretrained ResNet-18 weights into the Torch Hub
cache. Run this once before the first training run if torchvision's
automatic download fails with an SSL certificate error (common with the
python.org Python build on macOS, which ships without CA certificates).

Uses urllib with certifi's CA bundle; falls back to curl.
"""

import hashlib
import os
import subprocess
import sys
import urllib.request

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
SHA256_PREFIX = "f37072fd"


def main():
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, "resnet18-f37072fd.pth")

    if os.path.exists(dest):
        print(f"Already present: {dest}")
        return

    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=ctx) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as e:
        print(f"urllib download failed ({e}); trying curl")
        subprocess.check_call(["curl", "-sL", "-o", dest, URL])

    digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
    if not digest.startswith(SHA256_PREFIX):
        os.remove(dest)
        sys.exit(f"Hash mismatch (got {digest[:8]}), deleted {dest}")
    print(f"Downloaded and verified: {dest}")


if __name__ == "__main__":
    main()
