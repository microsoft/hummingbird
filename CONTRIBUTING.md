# Contributing

## Welcome

If you are here, it means you are interested in helping us out. A hearty welcome and thank you! There are many ways you can contribute to Hummingbird:

* Offer PR's to fix bugs or implement new features;
* Give us feedback and bug reports regarding the software or the documentation;
* Improve our examples, and documentation.
This project welcomes contributions and suggestions.

## Getting Started

Please join the community on Gitter *gitter badge*. Also please make sure to take a look at the project [roadmap](wiki/Roadmap-for-Upcoming-Features-and-Support).


### Pull requests
If you are new to GitHub [here](https://help.github.com/categories/collaborating-with-issues-and-pull-requests/) is a detailed help source on getting involved with development on GitHub.

As a first time contributor, you will be invited to sign the Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com. You will only need to do this once across all repos using our CLA.

Your pull request needs to reference a filed issue. Please fill in the template that is populated for the pull request. Only pull requests addressing small typos can have no issues associated with them.

All commits in a pull request will be [squashed](https://github.blog/2016-04-01-squash-your-commits/) to a single commit with the original creator as author.

### Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Developing
The simplest setup is:
```
mkdir hummingbird
cd hummingbird
git clone https://github.com/microsoft/hummingbird.git .
pip install -e .[docs,tests,extra]
```
On Windows, the last line above must also contain `-f https://download.pytorch.org/whl/torch_stable.html`. (This is required because PyTorch version for Windows is not up to date.)

### Docker
We provide a simple [Dockerfile](https://github.com/microsoft/hummingbird/blob/master/Dockerfile) that you can customize to your preferred development environment.
```
docker build git://github.com/microsoft/hummingbird -t hb-jupy
docker run -it hb-dev
```
### Codespases
For a light-weight, web-based experience, we provide the configuration ([.devcontainer](https://github.com/microsoft/hummingbird/tree/master/.devcontainer)) for [Codespaces](https://online.visualstudio.com/environments).  More information on this setup can be found [here]( https://docs.microsoft.com/en-us/visualstudio/online/reference/configuring).

### Tools
#### Pre-commit
This project uses [pre-commit](https://pre-commit.com/) hooks. Run  `pip install pre-commit` if you don't already have this in your machine. Afterward, run `pre-commit install` to install pre-commit into your git hooks.

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

#### Formatting
We generally use all pep8 checks, with the exception of line length 127.

To do a quick check-up before commit, try:
```
flake8 . --count  --max-complexity=10 --max-line-length=127 --statistics
```

#### Coverage

For coverage, we use [coverage.py](https://coverage.readthedocs.io/en/coverage-5.0.4/) in our Github Actions.  Run  `pip install coverage` if you don't already have this, and any code you commit should generally not significantly impact coverage.

We strive to not let check-ins decrease coverage.  To run all unit tests:
```
coverage run -m pytest tests
```
