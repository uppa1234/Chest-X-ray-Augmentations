import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    # Required argument: path
    parser.add_argument("path", help='Path to folder with picture files', type=str)
    # Optional arguments: size, output
    parser.add_argument("-s", "--size", help='Size to resize to', default=224, type=int)
    parser.add_argument("-o", "--output", help='Output extension', default='.png', type=str)
    args = parser.parse_args()
    og_path = Path(args.path)
    # Name of new folder
    new_dir = f'resized_{args.size}'
    print(f'Resizing images in {og_path} to {(args.size, args.size)} into {new_dir}.')

    # Create new folders
    for dir in og_path.glob("**"):
        if dir.is_dir():
            if new_dir not in str(dir):
                new_folder = og_path / new_dir / dir.relative_to(og_path)
                new_folder.mkdir(parents=True, exist_ok=True)

    # Locate target suffixes
    def get_paths(path, suffixes=['jpg', 'jpeg', 'png']):
        for suffix in suffixes:
            yield from path.rglob('*.{}'.format(suffix.strip('.')))
    
    paths = [p for p in get_paths(og_path) if new_dir not in str(p)]

    # Convert images
    for path in tqdm(paths):
        new_path = og_path / new_dir / path.relative_to(og_path)
        pic = Image.open(path).resize((args.size,args.size))
        pic.save(new_path.with_suffix(args.output))

if __name__ == "__main__":
    main()



