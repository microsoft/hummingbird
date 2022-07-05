# Documentation Website Generation Documentation

This document has some brief notes about how we turn our in-code documentation into flat html files for hosting via Github Pages.

## Tools

We use [`sphinx`](https://www.sphinx-doc.org/en/master/) to generate html documentation from the Python source code comments.

```
python -m pip install sphinx sphinx_rtd_theme
```

## Commands

This are run automatically in the pipeline on push:

```
make sphinx-site -C website/
```
