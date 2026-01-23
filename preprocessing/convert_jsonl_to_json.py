import json
import os


def convert_dir(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".jsonl"):
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.join(dirpath, name[:-6] + ".json")

            print(f"Converting {src} -> {dst}")
            data = []
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_data")
    convert_dir(base_dir)


