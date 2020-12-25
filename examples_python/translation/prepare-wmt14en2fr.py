import os
import subprocess
from pathlib import Path
import sys
import argparse

from loguru import logger


DESCRIPTION = """Adapted from
    https://github.com/hnt4499/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh
Prepare data for WMT14 en-fr machine translation task, including:
1. Clone `moses` and `subword-nmt` (if paths are not specified).
2. Download (and extract) raw data (train and test) (if paths not specified),
and save to "`save_dir`/orig".
3. Pre-process data and save to "`save_dir`/processed":
    i. Normalize punctuations.
    ii. Remove non-printing characters.
    iii. Tokenize
    iv. Split train data into train and val data.
4. Learn BPE from split training data and save to "`save_dir`/cleaned".
5. Apply BPE to train/val/test data (and temporarily save to
"`save_dir`/cleaned").
6. Remove sentence pairs that either:
    i. has source/target or target/source ratio > 1.5
    ii. has source or target sentence length < 1 or > 250
and save to "`save_dir`/cleaned"
"""


SRC_LANG = "en"
TGT_LANG = "fr"

URLS = [
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz",
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz",
    "http://statmt.org/wmt13/training-parallel-un.tgz",
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz",
    "http://statmt.org/wmt10/training-giga-fren.tar",
    "http://statmt.org/wmt14/test-full.tgz",
]

TRAIN_CORPORA = [
    # "europarl-v7.fr-en",
    # "commoncrawl.fr-en",
    # "undoc.2000.fr-en",
    "news-commentary-v9.fr-en",
    # "giga-fren.release2.fixed",
]
TEST_CORPUS = "newstest2014-fren-{}.{}.sgm"


def execute(command):
    logger.info(f"Executing: {command}")

    try:
        subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.info(
            f"An error occured, exiting...:\n{e.output.decode('utf-8')}")
        exit()


def remove_if_exists(path, ask=False):
    if os.path.exists(path):
        if ask:
            prompt = f"File existed: {path}. Overwrite? (y/n) "
            inp = input(prompt)
            while inp not in ["y", "n"]:
                print("Wrong input!")
                inp = input(prompt)
            if inp == "y":
                os.remove(path)
            else:
                return False
        else:
            os.remove(path)
    return True


def main(args):
    # Get logger
    logger.remove()  # remove default handler
    logger_path = os.path.join(args.save_dir, "log")
    logger.add(
        sys.stderr, colorize=True,
        format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {message}")
    logger.add(logger_path, mode="w",
               format="{time:YYYY-MM-DD at HH:mm:ss} | {message}")

    # Moses
    if args.moses_path is None or not os.path.isdir(args.moses_path):
        logger.info("Cloning Moses github repository to the current working "
                    "directory (for tokenization scripts)...")
        execute("git clone https://github.com/moses-smt/mosesdecoder.git")
        moses_path = os.path.realpath("./mosesdecoder/scripts")
    else:
        logger.info(f"Moses scripts found at {args.moses_path}")
        moses_path = os.path.realpath(args.moses_path)
    # Script paths
    tokenizer = os.path.join(moses_path, "tokenizer/tokenizer.perl")
    cleaner = os.path.join(moses_path, "training/clean-corpus-n.perl")
    norm_punc = os.path.join(
        moses_path, "tokenizer/normalize-punctuation.perl")
    rem_non_print_char = os.path.join(
        moses_path, "tokenizer/remove-non-printing-char.perl")

    # Subword NMT
    if args.subword_nmt is None or not os.path.isdir(args.subword_nmt):
        logger.info("Cloning Subword NMT repository to the current working "
                    "directory (for BPE pre-processing)...")
        execute("git clone https://github.com/rsennrich/subword-nmt.git")
        subword_nmt = os.path.realpath("./subword-nmt/subword_nmt")
    else:
        logger.info(f"Subword NMT scripts found at {args.subword_nmt}")
        subword_nmt = os.path.realpath(args.subword_nmt)

    save_dir = os.path.realpath(args.save_dir)
    # Original data
    if args.orig_dir is None or not os.path.isdir(args.orig_dir):
        orig_save_dir = os.path.join(save_dir, "orig")
        os.makedirs(orig_save_dir, exist_ok=True)
        logger.info(f"Original data not found. Downloading and extracting to "
                    f"{orig_save_dir}...")

        for url in URLS:
            # Download
            execute(f"wget {url} -P {orig_save_dir}")
            # Extract
            _, arc_filename = os.path.split(url)
            arc_filepath = os.path.join(orig_save_dir, arc_filename)
            if arc_filename.endswith(".tgz"):
                execute(f"tar xvzf {arc_filepath}")
            elif arc_filename.endswith(".tar"):
                # This is `giga-fren.release2.fixed`
                execute(f"tar xvf {arc_filepath}")
                # After extracting the tarball, we need to further extract
                # the gunzip
                for gunzip_path in Path("./").glob(
                        "giga-fren.release2.fixed.*.gz"):
                    execute(f"gunzip {gunzip_path}")
    else:
        orig_save_dir = os.path.realpath(args.orig_dir)
        logger.info(f"Original data found at {orig_save_dir}")

    # Preprocess train data
    processed_save_dir = os.path.join(save_dir, "processed")
    os.makedirs(processed_save_dir, exist_ok=True)
    logger.info(
        f"Pre-processing train data and save to {processed_save_dir}...")

    for lang in [SRC_LANG, TGT_LANG]:
        dest_filepath = os.path.join(
            processed_save_dir,
            f"train.unsplit.processed.{SRC_LANG}-{TGT_LANG}.{lang}")
        deleted = remove_if_exists(dest_filepath, ask=True)
        if not deleted:  # meaning we want to keep the existing file
            logger.info(f"File existed: {dest_filepath}. Skipping...")
            continue

        for raw_filename in TRAIN_CORPORA:
            raw_filepath = list(Path(orig_save_dir).rglob(
                f"{raw_filename}.{lang}"))[0]

            cmd = (
                f"cat {raw_filepath} | perl {norm_punc} {lang} | perl "
                f"{rem_non_print_char} | perl {tokenizer} --threads 8 -a -l "
                f"{lang} >> {dest_filepath}"
            )
            execute(cmd)

    processed_filepaths = {"train": [], "val": [], "test": []}

    # Preprocess test data
    for lang, source in [(SRC_LANG, "src"), (TGT_LANG, "ref")]:
        raw_filename = TEST_CORPUS.format(source, lang)
        raw_filepath = list(Path(orig_save_dir).rglob(raw_filename))[0]
        dest_filepath = (f"{processed_save_dir}/test."
                         f"{SRC_LANG}-{TGT_LANG}.{lang}")
        cmd = (
            f"grep '<seg id' {raw_filepath} | sed -e "
            f"'s/<seg id=\"[0-9]*\">\\s*//g' | sed -e 's/\\s*<\\/seg>\\s*//g' "
            f"| sed -e \"s/\\’/\\'/g\" | perl {tokenizer} -threads 8 -a -l "
            f"{lang} > {dest_filepath}"
        )
        execute(cmd)
        processed_filepaths["test"].append(dest_filepath)

    # Data splitting
    logger.info(f"Splitting train and valid data to {processed_save_dir}...")
    for lang in [SRC_LANG, TGT_LANG]:
        src_filepath = (f"{processed_save_dir}/train.unsplit.processed."
                        f"{SRC_LANG}-{TGT_LANG}.{lang}")
        dest_train_filepath = (f"{processed_save_dir}/train.processed."
                               f"{SRC_LANG}-{TGT_LANG}.{lang}")
        dest_val_filepath = (f"{processed_save_dir}/val.processed."
                             f"{SRC_LANG}-{TGT_LANG}.{lang}")

        cmd = ("awk '{if (NR%1333 == 0)  print $0; }' "
               f"{src_filepath} > {dest_val_filepath}")
        execute(cmd)
        processed_filepaths["val"].append(dest_val_filepath)

        cmd = ("awk '{if (NR%1333 != 0)  print $0; }' "
               f"{src_filepath} > {dest_train_filepath}")
        execute(cmd)
        processed_filepaths["train"].append(dest_train_filepath)

        # Remove `src_file`, which is unsplit data
        execute(f"rm {src_filepath}")

    # Merge two train files into one to learn BPE
    logger.info("Merging two train files into one to learn BPE from it...")
    merged_filepath = (f"{processed_save_dir}/train.processed."
                       f"{SRC_LANG}-{TGT_LANG}")
    remove_if_exists(merged_filepath)

    for processed_filepath in processed_filepaths["train"]:
        execute(f"cat {processed_filepath} >> {merged_filepath}")

    # Learn BPE
    cleaned_path = os.path.join(save_dir, "cleaned")
    os.makedirs(cleaned_path, exist_ok=True)

    code_path = os.path.join(cleaned_path, "code")
    logger.info(f"Learn BPE on {merged_filepath} and save code to {code_path}")
    cmd = (f"python {subword_nmt}/learn_bpe.py -s "
           f"{args.bpe_tokens} < {merged_filepath} > {code_path}")
    execute(cmd)

    # Apply BPE
    file_pairs = []
    for source, filepaths in processed_filepaths.items():
        for src_filepath in filepaths:
            src_filedir, src_filename = os.path.split(src_filepath)
            dest_filepath = os.path.join(src_filedir, f"bpe.{src_filename}")

            logger.info(f"Applying learned BPE code to {src_filepath}")
            cmd = (f"python {subword_nmt}/apply_bpe.py -c {code_path} < "
                   f"{src_filepath} > {dest_filepath}")
            execute(cmd)

            pair_filename, _ = os.path.splitext(dest_filepath)
            file_pairs.append(pair_filename)

    # Clean data
    logger.info("Cleaning data...")
    file_pairs_ = []
    for src_file_pair in set(file_pairs):
        _, file_pair = os.path.split(src_file_pair)
        dest_file_pair = os.path.join(  # remove the ".processed" part
            cleaned_path, file_pair.replace(".processed", ""))
        file_pairs_.append(dest_file_pair)

        # Don't clean the test data
        if "test" in src_file_pair:
            for lang in [SRC_LANG, TGT_LANG]:
                execute(f"cp {src_file_pair}.{lang} "
                        f"{dest_file_pair}.{lang}")

        else:
            cmd = (f"perl {cleaner} -ratio 1.5 {src_file_pair} {SRC_LANG} "
                   f"{TGT_LANG} {dest_file_pair} 1 250")
            execute(cmd)

    # Binarize data
    if args.binarize:
        logger.info("Binarizing data...")
        binarized_path = os.path.join(save_dir, "binarized")
        os.makedirs(binarized_path, exist_ok=True)
        train_pref, val_pref, test_pref = file_pairs_

        cmd = (
            f"fairseq-preprocess --source-lang {SRC_LANG} --target-lang "
            f"{TGT_LANG} --trainpref {train_pref} --validpref {val_pref} "
            f"--testpref {test_pref} --destdir {binarized_path} --workers 20"
        )
        execute(cmd)

    logger.info("Done")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        "-m", "--moses-path", type=str, required=False, default=None,
        help="Path to the `mosesdecoder/scripts` directory. Automatically "
             "clone from GitHub to the current working directory if not "
             "speficied.")
    parser.add_argument(
        "-s", "--subword-nmt", type=str, required=False, default=None,
        help="Path to the `subword-nmt/subword_nmt` directory. Automatically "
             "clone from GitHub to the current working directory if not "
             "speficied.")
    parser.add_argument(
        "-b", "--bpe-tokens", type=int, required=True,
        help="Number of BPE tokens to learn from the training data.")
    parser.add_argument(
        "--binarize", action="store_true", default=False,
        help="Whether to binarize data using `fairseq-preprocess`. Original "
             "preprocess script (`prepare-wmt14en2fr.sh`) does not include "
             "this.")

    parser.add_argument(
        "-o", "--orig-dir", type=str, required=False, default=None,
        help="Path to the original data (after extracted). Note that the data "
             "filenames must match with the original filename after extracted."
             " However, the data files do not necessarily need to be inside "
             "the top directory (i.e., it can be inside any subdirectories). "
             "Automatically download, extract and save to `save_dir`/orig if "
             "not specified.")
    parser.add_argument(
        "-d", "--save-dir", type=str, required=True,
        help="Directory to save the original as well as processed data. "
             "Subdirectories will be created inside this directory.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
