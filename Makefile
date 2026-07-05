PYTHON2 = ./_pypy_binary/bin/python2
RPYTHON = ./pypy/rpython/bin/rpython
RPYTHON_ARGS =

TARGET = targetrpyforth

.PHONY: build
build: build-jit build-jit-stkfrag build-jit-novirt

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
	wget -N https://www.complang.tuwien.ac.at/forth/gforth/Snapshots/current/gforth.tar.xz
	tar xJf gforth.tar.xz
	src=$$(tar tJf gforth.tar.xz | sed 's#/.*##' | grep -m1 '^gforth-'); \
	dest=$$(printf '%s' "$$src" | sed -E 's/_[0-9]{8}$$//'); \
	if [ "$$src" != "$$dest" ]; then rm -rf "$$dest"; mv "$$src" "$$dest"; fi; \
	echo "gforth source ready in $$dest (build it with: make build-gforth)"

_pypy_binary/bin/python:  ## Download a PyPy binary
	mkdir -p _pypy_binary
	python3 get_pypy_to_download.py
	tar -C _pypy_binary --strip-components=1 -xf pypy.tar.bz2
	rm pypy.tar.bz2
	./_pypy_binary/bin/python -m ensurepip
	./_pypy_binary/bin/python -mpip install "hypothesis<4.40" junit_xml coverage==5.5 "pdbpp==0.10.3"

GFORTH_DIR = gforth-0.7.9
GFORTH_FAST = ./$(GFORTH_DIR)/gforth-fast
GFORTH = ./$(GFORTH_DIR)/gforth

.PHONY: build-gforth
build-gforth: $(GFORTH_DIR)/gforth-fast

# Compile Gforth in-tree. The source tree is an order-only prerequisite so it is
# fetched on demand when absent (without forcing a rebuild when only its mtime
# changes). This keeps `make bench-shootout` self-bootstrapping instead of dying
# with "cd: $(GFORTH_DIR): No such file or directory" on a fresh checkout.
$(GFORTH_DIR)/gforth-fast: | $(GFORTH_DIR)
	cd $(GFORTH_DIR) && ./configure --prefix="$$PWD/_install" && $(MAKE) -j$$(nproc)

$(GFORTH_DIR):
	$(MAKE) setup-gforth

# run_shootout.py renders its --chart / --curve-chart PDFs with matplotlib.
# The system Python is externally managed (PEP 668), so a bare `pip install`
# is refused; install matplotlib into a project-local venv instead and run the
# benchmark driver from it. Create it once with `make setup-plot`.
VENV = .venv
PLOT_PY = $(VENV)/bin/python

$(PLOT_PY):
	python3 -m venv $(VENV)
	$(PLOT_PY) -m pip install --upgrade pip
	$(PLOT_PY) -m pip install matplotlib

.PHONY: setup-plot
setup-plot: $(PLOT_PY)

.PHONY: bench-shootout
bench-shootout: build-jit-stkfrag build-gforth $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_shootout.py \
    	--compare ./rpyforth-c-stkfrag --compare $(GFORTH_FAST) --compare $(GFORTH) \
    	--exclude curve/ --iterations 5 \
    	--chart compare.pdf

.PHONY: bench-shootout-curve
bench-shootout-curve: build-jit-stkfrag build-gforth $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_shootout.py \
    	--compare ./rpyforth-c-stkfrag --compare $(GFORTH_FAST) --compare $(GFORTH) \
    	--only curve/ --iterations 5 \
    	--curve-chart warmup.pdf

.PHONY: sweep-framesize
sweep-framesize: $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_param_sweep.py \
    	--iterations 3 --pin 2 \
    	--pdf sweep-framesize.pdf
