
clean:
	pip uninstall cnn -y
	#rm -r dist build

whl:
	python setup.py bdist_wheel

install: clean whl
	pip install -f dist cnn

run: clean whl install
	python test_script.py

