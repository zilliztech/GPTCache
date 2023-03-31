install:
	@pip install -r requirements.txt
	@python setup.py install

pip_upgrade:
	@python -m pip install --upgrade pip

package:
	@python setup.py sdist bdist_wheel

remove_example_cache.sh:
	@bash ./script/remove_example_cache.sh

create_conda_env:
	@bash ./script/manage_conda_env.sh create

remove_conda_env:
	@bash ./script/manage_conda_env.sh remove
