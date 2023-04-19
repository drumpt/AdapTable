"""
Utility script to remove any illegal characters  form columns names."""

import argparse
import glob
import os
import pandas as pd
import json
import copy

from tableshift.core.utils import sub_illegal_chars


def main(cache_dir, experiment, domain_split_varname, index_colname='Unnamed: 0'):
    glob_str = os.path.join(cache_dir, f"{experiment}domain_split_varname_{domain_split_varname}*")
    dirs = glob.glob(glob_str, recursive=False)
    print(f"[DEBUG] got dirs: {dirs}")
    for dir in dirs:
        for root, splitdirs, files in os.walk(dir):
            if splitdirs:
                print(f"[DEBUG] got split dirs {splitdirs}")
                for splitdir in splitdirs:
                    cache_files = glob.glob(os.path.join(dir, splitdir, "*.csv"))
                    print(f"cache_files: {cache_files}")
                    for cache_file in cache_files:
                        print(f"[INFO] processing file {cache_file}")
                        df = pd.read_csv(cache_file)
                        df.columns = [sub_illegal_chars(c) for c in df.columns]
                        print(f"[INFO] writing back to {cache_file}")
                        df.to_csv(cache_file, index=False)

                # update info.json
                info_fp = os.path.join(dir, "info.json")
                print(f"[INFO] updating info.json")
                with open(info_fp, "r") as f:
                    ds_info = copy.deepcopy(json.loads(f.read()))
                ds_info['target'] = sub_illegal_chars(ds_info['target'])
                ds_info['domain_label_colname'] = sub_illegal_chars(ds_info['domain_label_colname'])
                ds_info['group_feature_names'] = [sub_illegal_chars(c) for c in ds_info['group_feature_names']]
                ds_info['feature_names'] = [sub_illegal_chars(c) for c in ds_info['feature_names']]
                with open(info_fp, "w") as f:
                    f.write(json.dumps(ds_info))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./tmp", type=str)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--domain_split_varname", "-d", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
