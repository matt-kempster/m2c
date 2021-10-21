# mips_to_c
Given some MIPS assembly, this program will attempt to convert it to C.
The goal is that eventually the output will be well-formed C, and eventually after that, byte-equivalent C.

Right now the decompiler is fairly functional, though it sometimes generates suboptimal code
(especially for loops). See the `tests/` directory for some example input and output.

An online version is available at https://simonsoftware.se/other/mips_to_c.py.

## Install

This project requires Python 3.6 or later. To install the Python dependencies:
```bash
python3 -m pip install --upgrade pycparser

# Optional: If you are on python3.6, you will also need to install "dataclasses"
python3.6 -m pip install --upgrade dataclasses
```

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

Context files provided with `--context` are parsed and cached, so subsequent runs with the same file are faster. The cache for `foo/bar.c` is stored in `foo/bar.m2c`. Caching can be disabled with the `--no-cache` argument.

### Multiple functions

By default, `mips_to_c` decompiles all functions in the text sections from the input assembly files.
`mips_to_c` is able to perform a small amount of cross-function type inference, if the functions call each other.

You can limit the function(s) that decompiled by providing the `-f <function name>` flags (or the "Function" dropdown on the website).

### Global Declarations & Initializers

When provided input files with `data`, `rodata`, and/or `bss` sections, `mips_to_c` can generate the initializers for variables it knows the types of.

Qualifier hints such as `const`, `static`, and `extern` are based on which sections the symbols appear in, or if they aren't provided at all.
The output also includes prototypes for functions not declared in the context.

`mips_to_c` cannot generate initializers for structs with bitfields (e.g. `unsigned foo: 3;`) or for symbols that it cannot infer the type of.
For the latter, you can provide a type for the symbol the context.

This feature is controlled with the `--globals` option (or "Global declarations" on the website):

- `--globals=used` is the default behavior, global declarations are emitted for referenced symbols. Initializers are generated when the data/rodata sections are provided.
- `--globals=none` disables globals entirely; only function definitions are emitted.
- `--globals=all` includes all of the output in `used`, but also includes initializers for unreferenced symbols. This can be used to convert data/rodata files without decompiling any functions.

### Formatting

The following options control the formatting details of the output, such as braces style or numeric format. See `./mips_to_c.py --help` for more details. 

(The option name on the website, if available, is in parentheses.)

- `--valid-syntax`
- `--allman` ("Allman braces")
- `--pointer-style` ("`*` to the left")
- `--unk-underscore`
- `--hex-case`
- `--comment-style {multiline,oneline}` ("Comment style")
- `--comment-column N` ("Comment style")
- `--no-casts`

Note: `--valid-syntax` is used to produce output that is less human-readable, but is likely to directly compile without edits. This can be used to go directly from assembly to the permuter without human intervention.

### Debugging poor results (Advanced)

There are several options to `mips_to_c` which can be used to troubleshoot poor results. Many of these options produce more "primitive" output or debugging information.

- `--no-andor` ("Disable &&/||"): Disable complex conditional detection, such as `if (a && b)`. Instead, emit each part of the conditional as a separate `if` statement. Ands, ors, nots, etc. are usually represented with `goto`s.
- `--gotos-only` ("Use gotos for everything"): Do not detect loops or complex conditionals. This format is close to a 1-1 translation of the assembly.
    - Note: to use a goto for a single branch, don't use this flag, but add `# GOTO` to the assembly input.
- `--debug` ("Debug info"): include debug information inline with the code, such as basic block boundaries & labels.
- `--void` ("Force void return type"): assume that the decompiled function has return type `void`. Alternatively: provide the function prototype in the context.

#### Visualization

`mips_to_c` can generate an SVG representation of the control flow of a function, which can sometimes be helpful to untangle complex loops or early returns.

Pass `--visualize` on the command line, or use the "Visualize" button on the website. The output will be an SVG file.

Example to produce `my_fn.svg` of `my_fn()`:

```sh
python3 ./mips_to_c.py --visualize --context ctx.c -f my_fn my_asm.s > my_fn.svg
```

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
