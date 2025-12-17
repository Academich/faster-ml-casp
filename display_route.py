import json

from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.analysis.routes import RouteCollection

from tqdm import tqdm

def make_images(routes_json_path: str, num_of_molecules: int, output_dir: str):
    with open(routes_json_path, "r") as f:
        data = json.load(f)["data"]
        # data = [d for d in data if d["metadata"]["is_solved"]]

    tree_id = 0
    for d in tqdm(data):
        trees = [ReactionTree.from_dict(tree) for tree in d["trees"]]
        routes = RouteCollection(trees)

        images = routes.make_images()

        for i, image in enumerate(images):
            image.save(f"{output_dir}/{tree_id}_route_{i}.png")
        tree_id += 1
        if tree_id == num_of_molecules:
            break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_molecules", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--routes_json_path", "-i", type=str)
    args = parser.parse_args()

    make_images(args.routes_json_path, args.number_of_molecules, args.output_dir)



