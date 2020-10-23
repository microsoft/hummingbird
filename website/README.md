# Documentation Website Generation Documentation

This document has some brief notes about how we turn our in-code documentation into flat html files for hosting via Github Pages.

## Tools

We use [`sphinx`](https://www.sphinx-doc.org/en/master/) to generate html documentation from the Python source code comments.

```
pip install sphinx sphinx_rtd_theme
```

## Commands

These are run automatically in the pipeline on push.

To generate the .rst files:
```
sphinx-apidoc -o website/sphinx/api -t website/sphinx/_templates hummingbird/ml -d 1 -f -e
```

Then to run Sphinx and generate html:
```
make -C website/sphinx/
```
