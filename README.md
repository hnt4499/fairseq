<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
</p>

--------------------------------------------------------------------------------

Python wrapper of `fairseq`, a Sequence-to-Sequence toolkit written in Python by the Facebook AI Research Team. This wrapper is designed with the aim for greater flexibilty and configurability (especially for those who train models on Google Colab), while keeping backward compatibility with the original implementation. For more details on requirements, installation, etc., please refer to the [official implementation](https://github.com/pytorch/fairseq).

### Features:

#### New logging mechanism:
Nicely colorized logging mechanism with [loguru](https://github.com/Delgan/loguru) instead of old-school `logging`.
Additionally, one can simply add `--save-log` to direct the log stream to the current working directory (i.e., `save_dir`). 

#### `eval-bleu-print-samples`:
* Option to limit the number of samples to be printed in each validation step.
* No more annoyingly long logs during evaluation and avoid crashing when training with Google Colab!
* Just add `--eval-bleu-print-samples N`, where `N` is the number of sample pairs to be printed. Set `N=0` to completely suppress translation results.

#### `--val-suppress-progress-bar`:
* Option to complete suppress the progress bar during evaluation, which is not compatible with Google Colab.
* Just add `--val-suppress-progress-bar`, and the erroneous progress bar is gone!

#### `--val-log-interval`:
* Option to set the log interval duing evaluation. Used in conjunction with `--val-suppress-progress-bar`.\
* Just add `--val-log-interval N`, where `N` is the log interval.

#### ...and more to come!

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
