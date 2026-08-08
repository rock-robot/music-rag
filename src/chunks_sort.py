from pathlib import Path
import glob, os

def train_piece_names(augmented_train_dir):
    """Scrape the set of distinct piece names from the train-split filenames.

    Returns: a set of piece names, e.g. {'a_breeze_through_the_willows', ...}
    Raises if any filename lacks the '_idx' landmark.
    """
    # 1. list the .mid files in the directory
    files = sorted(glob.glob(os.path.join(augmented_train_dir, "*.mid")))
    # 2. for each: partition on "_idx"
    # 3. if the separator wasn't found -> that file is not what you think it is. Raise.
    # 4. collect the heads into a set
    train_pieces = set()
    for f in files:
        head, sep, tail = f.partition("_idx")
        if not sep:
            print(f + " does not have seperator")
        else:
            train_pieces.add(head)
        
    # 5. return the set
    return train_pieces

train_pieces = train_piece_names("data/train")   # adjust to your real path
print(len(train_pieces), "train pieces")
for name in sorted(train_pieces):
    print(" ", name)