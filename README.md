# Malpolon

## Installation

Currently, only the development version is available.
First make sure that the dependences listed in the `requirements.txt` file are installed.
`malpolon` can then be installed via `pip` using

```script
git clone https://github.com/plantnet/malpolon.git
cd malpolon
pip install -e .
```

## Examples

Examples using the GeoLifeCLEF 2022 dataset is provided in the `examples` folder.


## Documentation

To generate the documention, additional dependences contained in `docs/docs_requirements.txt` must be installed using

```script
pip install -r docs/docs_requirements.txt
```

The documentation can then be generated using

```script
make -C docs html
```

The result can be found in `docs/_build/html`.
