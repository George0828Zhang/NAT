import os
import re
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()

    data = {}

    header = ["src", "tgt"]

    for fold in args.source.split(','):
        tag = fold.strip('/').split('/')[-1]
        header.append(f"hypo-{tag}")

        filename = os.path.join(fold, "generate-test.txt")
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            src = None
            for line in reader:
                head = line[0]
                if re.match(r"^S-\d+$", head):
                    src = line[1]
                    if src not in data:
                        data[src] = {}
                elif re.match(r"^T-\d+$", head):
                    data[src]["tgt"] = line[1]
                elif re.match(r"^H-\d+$", head):                    
                    data[src][f"hypo-{tag}"] = line[2]
                elif re.match(r"^[DP]-\d+$", head) or "| INFO |" in head:
                    pass
                else:
                    raise RuntimeError(line)

    with open(args.output, "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for src, t in data.items():
            print(src, t)
            rowdata = [src] + [ t[tag] for tag in header[1:]]
            writer.writerow(rowdata)

    print("done!")
