import json
import os

import commentjson

from source.helpers import converter

if __name__ == '__main__':

    dir = "./low_lr"

    for file in os.listdir(dir):
        file_path = os.path.abspath(os.path.join(dir, file))
        if file_path.endswith(".jsonc"):
            with open(file_path, 'r+') as f:
                data = commentjson.load(f)

                # remove old parameter
                del data["training"]["general"]["log_to_comet"]
                # add new parameter
                data["training"]["general"]["comet"] = {"log_to_comet": False, "tags": ["augmentation", "low_lr"]}

                f.seek(0)  # reset file position
                json.dump(data, f, indent=4)
                f.truncate()  # remove remaining file content
