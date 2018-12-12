# mips_to_c
Given some MIPS assembly, this program will attempt to convert it to C.
The goal is that eventually the output will be well-formed C, and eventually after that, byte-equivalent C.

Right now the decompiler is fairly functional, though it sometimes generates suboptimal code
(especially for loops), and sometimes crashes. See the `tests/` directory for some example output.

## Install

Make sure you have Python 3.6 or later installed, then do `python3 -m pip install attrs`.

You might need to install `pip` first; on Ubuntu this can be done with:
```bash
sudo apt update
sudo apt install python3-pip
```

## Usage

```bash
python3 src/main.py [options] <asmfile> <functionname | index>
```

Run with `--help` to see which options are available.

## Contributing

There is much low-hanging fruit still. Take a look at the issues if you want to help out.

There is a small test suite, which works as follows: for every commit, `./run-tests.sh` should be run,
which runs the decompiler on a small corpus of IRIX 5.3-compiled MIPS assembly.
Any decompilations whose results change should be manually inspected with `git diff`
and committed along with the rest of the changes.

Type annotations are used for all Python code. `mypy src/main.py` should pass without any errors.
