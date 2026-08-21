# User Guide

## Installation and running the program

The project requires Python >= 3.12 and Poetry >= 2.0. Run the following commands in the root of the project folder.

Install the dependencies:

```python
poetry install
```

To run the program:

```python
poetry run python src/cli.py --num-components 20
```

`--num-components` controls how many eigenfaces are kept (default is 20).

## Tests

To run tests:

```python
poetry run pytest src
```
