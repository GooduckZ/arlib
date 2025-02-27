## Arlib 

Arlib is toolkit for automated reasoning. It provides a set of tools for constraint solving, logical inference, and symbolic computation.


## Local Development Environment

Run the following command to setup the local development environment.
~~~~
bash setup_local_env.sh
~~~~


The script will 
- Create a Python virtual environment if it doesn't exist
- Activate the virtual environment and install dependencies from requirements.txt
- Download required solver binaries (cvc5, mathsat, z3)
- Run unit tests if available

TBD:
- Test the scripts on different platforms, editors/IDEs, etc.

## Install the Library Locally

Local installziation via setup.py
~~~~
pip install -e .
~~~~

Then you can use a few cli tools of this library,
 add call the Python API in your own Python code.

## Release the Repo to PyPI

TBD (The repository is not yet released to PyPI.)

## Contributing

Contributions are welcome. Please refer to the repository for detailed instructions on how to contribute. 

~~~~
arlib/
├── arlib/           # Main library code
├── benchmarks/      # Benchmark files and test cases
├── bin_solvers/     # Binary solver executables
├── docs/            # Documentation files
├── scripts/         # Utility scripts
├── examples/        # A few applications
├── setup.py         # Package setup configuration (not ready)
├── pytest.ini       # PyTest configuration
└── requirements.txt # Project dependencies
~~~~

For Summer Research, Final Year Project Topics, please refer to
`docs/topics.rst` or `TODO.md`.

## Documentation

We release the docs here:
https://pyarlib.readthedocs.io/en/latest/


## Publications

Here are some of publications that use Arlib.

- [Enabling Runtime Verification of Causal Discovery Algorithms with Automated Conditional Independence Reasoning](https://arxiv.org/pdf/2309.05264.pdf)
Pingchuan Ma, Zhenlan Ji, Peisen Yao, Shuai Wang, and Kui Ren. ICSE 2024


## Contributors

Primary contributors to this project:
- rainoftime / cutelimination
- JasonJ2021
- ZelinMa557 
- Harrywwq
- little-d1d1
- ljcppp