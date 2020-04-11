import re
import sys
if __name__ == "__main__":
    pat = """    import wandb
    wandb.init(project="nat", sync_tensorboard=True)"""
    head = "def main("
    with open(sys.argv[1], "r") as f:
        buf = f.read()

    if pat in buf:
        print("injection already complete!")
        exit(0)

    newbuf = []
    for line in buf.split("\n"):
        newbuf.append(line)
        if head in line:
            newbuf.append(pat)
    
    if len(sys.argv) > 2:
        fname = sys.argv[2]
    else:
        fname = sys.argv[1]

    with open(fname, "w") as f:
        f.write("\n".join(newbuf))
