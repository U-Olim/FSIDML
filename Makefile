.PHONY: reproduce full clean

reproduce:
	pixi run pytask

full:
	pixi run pytask

clean:
	python -c "import shutil; shutil.rmtree('bld', ignore_errors=True)"
