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
	@bash ./script/remove_example_cache.sh

create_conda_env:
	@bash ./script/manage_conda_env.sh create

remove_conda_env:
	@bash ./script/manage_conda_env.sh remove
