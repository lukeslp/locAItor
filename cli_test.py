#!/usr/bin/env python3
"""
CLI Test Suite for /analyze Endpoint

Usage:
  python cli_test.py --image path/to/image.jpg [--host http://127.0.0.1:5002]
  python cli_test.py --dir path/to/images/ [--host http://127.0.0.1:5002]

Requires: requests
"""
import os
import sys
import argparse
import requests


def test_image(image_path, host):
    url = f"{host.rstrip('/')}/analyze"
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f)}
        try:
            resp = requests.post(url, files=files)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] {image_path}: {e}")
            return False
        print(f"\n=== {os.path.basename(image_path)} ===")
        if 'error' in data:
            print(f"[API ERROR] {data['error']}")
            return False
        print(f"[LLM Output]\n{data.get('content','')}")
        print(f"[AI Generated]: {data.get('metadata',{}).get('ai_generated')} | Reason: {data.get('metadata',{}).get('ai_reason')}")
        if 'metadata' in data:
            print("[Metadata keys]:", list(data['metadata'].keys()))
        return True


def main():
    parser = argparse.ArgumentParser(description="CLI test for /analyze endpoint.")
    parser.add_argument('--image', type=str, help='Path to a single image file')
    parser.add_argument('--dir', type=str, help='Directory of images to test')
    parser.add_argument('--host', type=str, default='http://127.0.0.1:5002', help='API host URL')
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error('Must specify --image or --dir')

    images = []
    if args.image:
        images = [args.image]
    elif args.dir:
        for fname in os.listdir(args.dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif')):
                images.append(os.path.join(args.dir, fname))
    if not images:
        print('No images found to test.')
        sys.exit(1)

    results = []
    for img in images:
        ok = test_image(img, args.host)
        results.append((img, ok))
    print(f"\nTested {len(results)} images. Success: {sum(ok for _,ok in results)} | Fail: {sum(not ok for _,ok in results)}")

if __name__ == '__main__':
    main() 