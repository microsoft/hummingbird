# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Developing

This project uses [pre-commit](https://pre-commit.com/) hooks. (Run  `pip install pre-commit` if you don't already have this.)

To begin, run `pre-commit install` to install pre-commit into your git hooks.

And before you commit, you can run it like this `pre-commit run --all-files` and should see output such as:

```
black............................Passed
Flake8...........................Passed
...
Don't commit to branch...........Passed
```

If you have installed your pre-commit hooks successfully, you should see something like this if you try to commit something non-conformant:
```
$ git commit -m "testing"
black............................Failed
- hook id: black
- files were modified by this hook

reformatted hummingbird/convert.py
All done!
1 file reformatted.
```

### Formatting
We generally use all pep8 checks, with the exception of line length 127.

To do a quick check-up before commit, try:
```
flake8 . --count  --max-complexity=10 --max-line-length=127 --statistics
```

### Coverage

For coverage, we use [coverage.py](https://coverage.readthedocs.io/en/coverage-5.0.4/) in our Github Actions. (`pip install coverage`)

We strive to keep our test coverage about 70%.  To run all unit tests:
```
coverage run -m pytest tests
```