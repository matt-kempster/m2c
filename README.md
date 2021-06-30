# mips_to_c
Given some MIPS assembly, this program will attempt to convert it to C.
The goal is that eventually the output will be well-formed C, and eventually after that, byte-equivalent C.

Right now the decompiler is fairly functional, though it sometimes generates suboptimal code
(especially for loops). See the `tests/` directory for some example input and output.

An online version is available at https://simonsoftware.se/other/mips_to_c.py.

## Install

Make sure you have Python 3.6 or later installed, then do `python3 -m pip install --upgrade pycparser` (also `dataclasses` if not on 3.7+).

You might need to install `pip` first; on Ubuntu this can be done with:
```bash
sudo apt update
sudo apt install python3-pip
```

## Usage

```bash
python3 mips_to_c.py [options] [--context <context file>] [-f <function name>] <asmfile>...
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

Type annotations are used for all Python code. `mypy mips_to_c.py` should pass without any errors.

To get pretty graph visualizations, install `graphviz` using `pip` and globally on your system (e.g. `sudo apt install graphviz`), and pass the `--visualize` flag.

## Tests

There is a small test suite, which works as follows:
 - As you develop your commit, occasionally run `./run_tests.py` to see if any tests have changed output.
   These tests run the decompiler on a small corpus of IDO 5.3-compiled MIPS assembly.
 - Before pushing your commit, run `./run_tests.py --overwrite` to write changed tests to disk, and commit resultant changes.

### Running Decompilation Project Tests

It's possible to use the entire corpus of assembly files from decompilation projects as regression tests.

For now, the output of these tests are not tracked in version control.
You need to run `./run_tests.py --overwrite ...` **before** making any code changes to create the baseline output.

As an example, if you have the `oot` project cloned locally in the parent directory containing `mips_to_c`, the following will decompile all of its assembly files.

```bash
./run_tests.py --project ../oot --project-with-context ../oot
```

This has been tested with:
- [zeldaret/oot](https://github.com/zeldaret/oot)
- [zeldaret/mm](https://github.com/zeldaret/mm)
    - See notes below, the repository needs to be partially built
- [pmret/papermario](https://github.com/pmret/papermario)
    - Need to use the `ver/us` or `ver/jp` subfolder, e.g. `--project ../papermario/ver/us`

#### Creating Context Files

The following bash can be used in each decompilation project to create a "universal" context file that can be used for decompiling any assembly file in the project.
This creates `ctx.c` in the project directory.

```bash
cd mm       # Or oot, papermario, etc.
find include/ src/ -type f -name "*.h" | sed -e 's/.*/#include "\0"/' > ctx_includes.c
tools/m2ctx.py ctx_includes.c
```

#### Notes for Majora's Mask

The build system in the MM decompilation project is currently being re-written.
It uses "transient" assembly that is not checked in, and in the normal build process it re-groups `.rodata` sections by function.

To use the MM project, run the following to *just* build the transient assembly files (and avoid running `split_asm.py`).

```bash
cd mm
make distclean
make setup
make asm/disasm.dep
```

The repository should be setup correctly if there are `asm/code`, `asm/boot`, and `asm/overlays` folders with `.asm` files, but there *should not* be an `asm/non_matchings` folder.

### Coverage

Code branch coverage can be computed by running `./run_tests.py --coverage`.
By default, this will generate an HTML coverage report `./htmlcov/index.html`.

### Adding an End-to-End Test

You are encouraged to add new end-to-end tests using the `./tests/add_test.py` script.

You'll need the IDO `cc` compiler and [sm64tools](https://github.com/queueRAM/sm64tools).

A good reference test to base your new test on is [`array-access`](tests/end_to_end/array-access).

Create a new directory in `tests/end_to_end`, and write the `orig.c` test case.
If you want the test to pass in C context, also add `irix-o2-flags.txt` & `irix-g-flags.txt` files.

After writing these files, run `add_test.py` with the path to the new `orig.c` file, as shown below.
This example assumes that sm64tools is cloned & built in your home directory, and that the IDO compiler is available from the OOT decompilation project.
You should change these exported paths to match your system.

```bash
export SM64_TOOLS=$HOME/sm64tools/build/
export IDO_CC=$HOME/oot/tools/ido_recomp/linux/7.1/cc
./tests/add_test.py $PWD/tests/end_to_end/my-new-test/orig.c
```

This should create `irix-o2.s` and `irix-g.s` files in your test directory.

Now, run `./run_tests.py --overwrite` to invoke the decompiler and write the output to `irix-o2-out.c` and `irix-g-out.c`. 
Finally, `git add` your test to track it.

```bash
./run_tests.py --overwrite
git add tests/end_to_end/my-new-test
```
