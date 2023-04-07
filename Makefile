install:
	@pip install -r requirements.txt
	@python setup.py install

pip_upgrade:
	@python -m pip install --upgrade pip

package:
	@python setup.py sdist bdist_wheel

upload:
	@python -m twine upload dist/*

upload_test:
	@python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

remove_example_cache:
	@bash ./scripts/remove_example_cache.sh

create_conda_env:
	@bash ./scripts/manage_conda_env.sh create

remove_conda_env:
	@bash ./scripts/manage_conda_env.sh remove

docs_build:
	cd docs && poetry run make html

docs_clean:
	cd docs && poetry run make clean

docs_linkcheck:
	poetry run linkchecker docs/_build/html/index.html

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$')

lint lint_diff:
	poetry run mypy $(PYTHON_FILES)
	poetry run black $(PYTHON_FILES) --check
	poetry run ruff .

pylint_check:
	pylint --rcfile=pylint.conf --output-format=colorized gptcache && pylint --rcfile=pylint.conf --output-format=colorized tests.unit_tests