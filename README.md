# Web doc branch for Hummingbird

To refresh the web documentation run `git subtree push --prefix doc/html/hummingbird origin gh-pages` from the `master` branch. 

The doc directory in `master` will be used as staging.

To generate the docs, run `pdoc3 hummingbird --html --force --template-dir=doc/pdoc_template/ --output-dir=doc/html`.
