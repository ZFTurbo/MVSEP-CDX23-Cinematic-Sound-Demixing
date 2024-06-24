# MVSEP-CDX23-Cinematic-Sound-Demixing

Model for [Sound demixing challenge 2023: Cinematic Sound Demixing Track - CDX'23](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023). Model performs separation of music into 3 stems "dialog (speech)", "effect (sfx)", "music". Model was trained on [DnR dataset](https://zenodo.org/record/5574713).  It based on [Demucs4](https://github.com/facebookresearch/demucs). Test set in CDX23 contest was very different from DnR train data. So we released best models with best metrics for DnR test set. 

## Usage

```
    python inference.py --input_audio mixture1.wav mixture2.wav --output_folder ./results/
```

With this command audios with names "mixture1.wav" and "mixture2.wav" will be processed and results will be stored in `./results/` folder in WAV format.

* **Note**: for slightly better quality of results use `--high_quality` parameter. It will be ~3 times slower.

## Quality metrics

Quality were measured on [DnR test set](https://zenodo.org/record/5574713)

| Algorithm     | SDR dialog  | SDR effect  | SDR music  | SDR mean |
| ------------- |:---------:|:----------:|:----------:|:----------:|
| Demucs HT 4 (single model)   | 14.18   | 7.92    | 6.75     | 9.62     |
| Demucs HT 4 (3 checkpoints ensemble)   | 14.68   | 8.48    | 7.30     | 10.16     |

* Note 1: SDR - signal to distortion ratio. Larger is better.
* Note 2: Music stem in DnR dataset can contain vocals

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

* [TISMIR paper](https://doi.org/10.5334/tismir.172)

```
@article{f2024sound,
    title={The Sound Demixing Challenge 2023 – Cinematic Demixing Track},
    author={Stefan Uhlich, Giorgio Fabbro, Masato Hirano, Shusuke Takahashi, Gordon Wichern, Jonathan Le Roux, Dipam Chakraborty, Sharada Mohanty, Kai Li, Yi Luo, Jianwei Yu, Rongzhi Gu, Roman Solovyev, Alexander Stempkovskiy, Tatiana Habruseva, Mikhail Sukhovei, Yuki Mitsufuji},
    volume={7},
    number={1},
    pages={44--62},
    year={2024}
}
```
