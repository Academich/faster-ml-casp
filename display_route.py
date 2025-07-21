import json

from aizynthfinder.reactiontree import ReactionTree
from aizynthfinder.analysis.routes import RouteCollection

def make_images(routes_json_path: str, index: int, output_dir: str):
    with open(routes_json_path, "r") as f:
        data = json.load(f)["data"]
        # data = [d for d in data if d["metadata"]["is_solved"]]

    for d in data:
        trees = [ReactionTree.from_dict(tree) for tree in d["trees"]]
        routes = RouteCollection(trees)

        images = routes.make_images()

        for i, image in enumerate(images):
            image.save(f"{output_dir}/route_{i}.png")
        break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--routes_json_path", "-i", type=str)
    args = parser.parse_args()

    make_images(args.routes_json_path, args.index, args.output_dir)



