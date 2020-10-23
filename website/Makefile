# A simple Makefile to help build all of the web page documentation from the
# source code and repo markdown docs.

# Make sure to check the usual pip install location for commands.
PATH := $(PATH):${HOME}/.local/bin
export PATH

.PHONY:
all: public/python_api


.PHONY:
clean: clean-sphinx-site


.PHONY:
clean-sphinx-site:
	rm -rf sphinx/_build
	rm -rf sphinx/api
	git checkout sphinx/api/hummingbird.rst
	rm -rf public/python_api

# Note: this is an expensive target.
.PHONY:
sphinx-site: #clean-sphinx-site
	pip install sphinx sphinx_rtd_theme psutil
	sphinx-apidoc -o sphinx/api -t sphinx/_templates ../hummingbird  -d 1 -f -e
	#running out of ideas on why this breaks things:
	rm sphinx/api/hummingbird.ml.operator_converters.onnx.onnx_operator.rst
	rm sphinx/api/hummingbird.ml.operator_converters.sklearn.iforest.rst
	rm sphinx/api/hummingbird.ml.operator_converters.sklearn.imputer.rst
	rm sphinx/api/hummingbird.ml.operator_converters.sklearn.pipeline.rst
	$(MAKE) -C sphinx
	$(MAKE) copy-python-apis

sphinx/_build/html:
	$(MAKE) sphinx-site

.PHONY:
copy-python-apis: sphinx/_build/html
	mkdir -p public
	rm -rf public/python_api
	cp -a sphinx/_build/html public/python_api

public/python_api:
	$(MAKE) copy-python-apis


