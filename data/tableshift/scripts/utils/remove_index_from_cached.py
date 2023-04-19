"""
Utility script to convert from 'old' cached data format to a new one
without re-caching the dataset."""

import argparse
import glob
import os
import re
import pandas as pd


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
                        if df.columns[0] == index_colname:
                            print(f"[DEBUG] dropping columns {index_colname} from {cache_file}")
                            df.drop(columns=[index_colname], inplace=True)
                            print(f"[INFO] writing back to {cache_file}")
                            df.to_csv(cache_file, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./tmp", type=str)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--domain_split_varname", "-d", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
