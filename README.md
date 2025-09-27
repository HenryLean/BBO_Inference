# BBO Inference

This repository provides an implementation of the algorithms proposed in the preprint paper:

**Black-box Optimization with Simultaneous Statistical Inference for Optimal Performance** 
> Teng Lian, Jian-Qiang Hu, Yuhang Wu, Zeyu Zheng  
> [arXiv:2501.07795](https://arxiv.org/abs/2501.07795)

The goal of this project is to reproduce the results and make the method more accessible for further research and applications.


## Installation

Clone the repository:

```bash
git clone https://github.com/HenryLean/BBO_Inference.git
cd BBO_Inference
```

## Usage

Run an experiment:

```bash
python main.py --alg_config ./_configs/alg_config_default2.json 
```

for `main.py`, or 

```bash
python toy_test.py --alg_config ./_configs/alg_config_default2.json 
```

By default the output is saved in the directory "./_cache/".

We also provide a script to 
- compute the mean and standard deviation of the results across replications and saves them into new CSV files;
- extract specific columns for histogram plotting;
- collect the run time data to a new location for easier access.

```bash
python xform.py
```

## Project Structure

```
BBO_Inference/
├── _configs/          # Experiment configuration files
├── src/               # Source code for algorithms
├── tests/             # Shell scripts for automated runs
├── main.py            # Main script
├── toy_test.py        # Test script for toy examples
├── xform.py           # Utility script for result analysis
└── README.md
```


## Citation 
If you find this implementation useful, please consider citing the original paper:

```bibtex
@misc{lian2025bbossiop,
    title={Black-box Optimization with Simultaneous Statistical Inference for Optimal Performance}, 
    author={Teng Lian and Jian-Qiang Hu and Yuhang Wu and Zeyu Zheng},
    year={2025},
    eprint={2501.07795},
    archivePrefix={arXiv},
    primaryClass={stat.CO},
    url={https://arxiv.org/abs/2501.07795}, 
}
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub.