# A simple Makefile to help build all of the web page documentation from the
# source code and repo markdown docs.

# Make sure to check the usual pip install location for commands.
PATH := $(PATH):${HOME}/.local/bin
export PATH

.PHONY:
all: public


.PHONY:
clean: clean-sphinx-site


.PHONY:
clean-sphinx-site:
	rm -rf sphinx/_build
	rm -rf sphinx/api
	rm -rf public

# Note: this is an expensive target.
.PHONY:
sphinx-site: #clean-sphinx-site
	pip install sphinx sphinx_rtd_theme psutil
	# TODO: There is a bug with torch 1.6 causing a "has no attribute 'training')" error.  Using 1.5 for now.
	pip install torch==1.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
	sphinx-apidoc --private -o sphinx/api -t sphinx/_templates ../hummingbird  -d 1 -f -e
	$(MAKE) -C sphinx
	$(MAKE) copy-python-apis

sphinx/_build/html:
	$(MAKE) sphinx-site

.PHONY:
copy-python-apis: sphinx/_build/html
	mkdir -p public
	rm -rf public
	cp -a sphinx/_build/html public
	cp -a ../website/images public/images

public:
	$(MAKE) copy-python-apis


