# Documentation Website Generation Documentation

This document has some brief notes about how we turn our in-code documentation into flat html files for hosting via Github Pages.

## Tools

We use [`sphinx`](https://www.sphinx-doc.org/en/master/) to generate html documentation from the Python source code comments.

## Commands

- [`sphinx/apidoc.sh`](./sphinx/apidoc.sh)
  - prepares for using `sphinx` to generate documentation for the Python APIs from the code comments.
