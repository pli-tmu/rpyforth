PYTHON2 = ./_pypy_binary/bin/python2
RPYTHON = ./pypy/rpython/bin/rpython
RPYTHON_ARGS =

TARGET = targetrpyforth

.PHONY: build
build: build-jit

.PHONY: setup-pypy
setup-pypy:
	if [ ! -d pypy ]; then git clone https://github.com/pypy/pypy.git --depth=1; fi

.PHONY: build-interp
build-interp: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. $(PYTHON2) $(RPYTHON) -O2 $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit
build-jit: _pypy_binary/bin/python setup-pypy
	RPYFORTH_VIRTUALIZE=1 PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag
build-jit-stkfrag: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_FRAGMENT=1 PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-novirt
build-jit-novirt: _pypy_binary/bin/python setup-pypy
	RPYFORTH_EXE_NAME=rpyforth-c-novirt PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-all
build-all: build-jit build-jit-novirt

.PHONY: test
test: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyforth/test -vv -s

.PHONY: coverage
coverage:
	python3 check_coverage.py

.PHONY: setup-gforth
setup-gforth:
	wget https://www.complang.tuwien.ac.at/forth/gforth/Snapshots/current/gforth.tar.xz
	tar xvfJ gforth.tar.xz
	cd gforth-*
	./install-deps.sh
	./configure
	make
	sudo make install

_pypy_binary/bin/python:  ## Download a PyPy binary
	mkdir -p _pypy_binary
	python3 get_pypy_to_download.py
	tar -C _pypy_binary --strip-components=1 -xf pypy.tar.bz2
	rm pypy.tar.bz2
	./_pypy_binary/bin/python -m ensurepip
	./_pypy_binary/bin/python -mpip install "hypothesis<4.40" junit_xml coverage==5.5 "pdbpp==0.10.3"
