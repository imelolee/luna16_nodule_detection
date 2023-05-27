# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import csv
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=["./trained_results/result_luna16_fold0.json"],
        help="input json",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./trained_results/result_luna16_fold0.csv",
        help="output csv",
    )

    args = parser.parse_args()

    in_json_list = args.input
    out_csv = args.output

    with open(out_csv, "w", newline="") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability"])
        for in_json in in_json_list:
            result = json.load(open(in_json, "r"))
            for subj in result["validation"]:
                if subj['image'].endswith(".npy"):
                    seriesuid = os.path.split(subj["image"])[-1][:-4]
                    for b in range(len(subj["box"])):
                        bboxes = subj["box"][b][0:3]
                        bboxes.reverse()
                        spamwriter.writerow([seriesuid] + bboxes + [subj["score"][b]])
                if subj['image'].endswith(".nii.gz"):
                    seriesuid = os.path.split(subj["image"])[-1][:-7]
                    for b in range(len(subj["box"])):
                        spamwriter.writerow([seriesuid] + subj["box"][b][0:3] + [subj["score"][b]])


if __name__ == "__main__":
    main()
