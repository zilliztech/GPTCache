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

pylint_check:
	pylint --rcfile=pylint.conf --output-format=colorized gptcache

pytest:
	pytest tests/