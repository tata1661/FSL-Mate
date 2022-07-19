# SOURCE_GLOB=$(wildcard paddlefsl/**/*.py tests/**/*.py examples/*.py)
# TODO: add more glob source files
SOURCE_GLOB=$(wildcard PaddleFSL/paddlefsl/__init__.py)

IGNORE_PEP=E203,E221,E241,E272,E501,F811

export PYTHONPATH=PaddleFSL


.PHONY: all
all : clean lint

.PHONY: clean
clean:
	rm -fr dist/* .pytype ./build/ dist/

.PHONY: lint
lint: pylint pycodestyle flake8 mypy


.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		--disable=W0511,R0801,cyclic-import,C4001 \
		$(SOURCE_GLOB)

.PHONY: pycodestyle
pycodestyle:
	pycodestyle \
		--statistics \
		--count \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: flake8
flake8:
	flake8 \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: mypy
mypy:
	MYPYPATH=stubs/ mypy \
		$(SOURCE_GLOB)

.PHONY: pytype
pytype:
	pytype \
		-V 3.8 \
		--disable=import-error,pyi-error \
		src/
	pytype \
		-V 3.8 \
		--disable=import-error \
		examples/

.PHONY: install
install:
	pip3 install -r requirements.txt
	pip3 install -r requirements-dev.txt

.PHONY: pytest
pytest:
	pytest src/ test/

.PHONY: test
test: 
	make pylint

.PHONY: dist
dist:
	make clean
	python3 setup.py sdist bdist_wheel

.PHONY: publish
publish:
	PATH=~/.local/bin:${PATH} twine upload dist/*

.PHONY: version
version:
	@newVersion=$$(awk -F. '{print $$1"."$$2"."$$3+1}' < VERSION) \
		&& echo $${newVersion} > VERSION \
		&& git add VERSION \
		&& git commit -m "🔥 update version to $${newVersion}" > /dev/null \
		&& git tag "v$${newVersion}" \
		&& echo "Bumped version to $${newVersion}"

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > paddlefsl/version.py

.PHONY: doc
doc:
	mkdocs serve
