
clean:
	pip uninstall cnn -y
	#rm -r dist build

whl:
	python setup.py bdist_wheel

run: clean whl
	pip install -f dist cnn
	python test_script.py

