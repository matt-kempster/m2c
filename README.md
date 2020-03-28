# mips_to_c
Given some MIPS assembly, this program will attempt to convert it to C.
The goal is that eventually the output will be well-formed C, and eventually after that, byte-equivalent C.

Right now the decompiler is fairly functional, though it sometimes generates suboptimal code
(especially for loops), and sometimes crashes. See the `tests/` directory for some example output.

## Install

Make sure you have Python 3.6 or later installed, then do `python3 -m pip install --upgrade attrs pycparser`.

You might need to install `pip` first; on Ubuntu this can be done with:
```bash
sudo apt update
sudo apt install python3-pip
```

## Usage

```bash
python3 mips_to_c.py [options] <asmfile> <functionname | index | all>
```

Run with `--help` to see which options are available.

## Contributing

There is much low-hanging fruit still. Take a look at the issues if you want to help out.

We use `black` to auto-format our code. We recommend using `pre-commit` to ensure only auto-formatted code is committed. To set these up, run:
```bash
pip install pre-commit black
pre-commit install
```

Your commits will then be automatically formatted per commit. You can also manually run `black` on the command-line.

There is a small test suite, which works as follows:
 - As you develop your commit, occasionally run `./run_tests.py` to see if any tests have changed output.
   These tests run the decompiler on a small corpus of IRIX 5.3-compiled MIPS assembly.
 - Before pushing your commit, run `./run_tests.py --overwrite` to write changed tests to disk, and commit resultant changes.

You are encouraged to add new tests using the `./tests/add_test.py` script.
Make sure to `./run_tests.py` after adding new tests.

Type annotations are used for all Python code. `mypy mips_to_c.py` should pass without any errors.

To get pretty graph visualizations, install `graphviz` using `pip` and globally on your system (e.g. `sudo apt install graphviz`), and pass the `--visualize` flag.
