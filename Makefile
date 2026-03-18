.PHONY: reproduce fast full clean

reproduce:
	pixi run pytask

fast:
	pixi run pytask-fast

full:
	pixi run pytask

clean:
	python -c "import shutil; shutil.rmtree('bld', ignore_errors=True)"
