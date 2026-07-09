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
build-all: build-jit built-jit-stkfrag build-jit-novirt

.PHONY: test
test: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyforth/test -vv -s

.PHONY: test-joy
test-joy: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyjoy/test -q

.PHONY: bench-joy
bench-joy: _pypy_binary/bin/python setup-pypy
	.venv/bin/python benchmark/run_joy.py --iterations 3

.PHONY: build-joy
build-joy: _pypy_binary/bin/python setup-pypy
	RPYJOY_EXE_NAME=rpyjoy-c PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyjoy/targetrpyjoy.py

.PHONY: build-joy-stkfrag
build-joy-stkfrag: _pypy_binary/bin/python setup-pypy
	RPYJOY_STACK_FRAGMENT=1 RPYJOY_EXE_NAME=rpyjoy-c-stkfrag PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyjoy/targetrpyjoy.py

.PHONY: coverage
coverage:
	python3 check_coverage.py

.PHONY: setup-gforth
setup-gforth: download-gforth build-gforth

.PHONY: download-gforth
download-gforth:
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

$(GFORTH_DIR)/gforth-fast: | $(GFORTH_DIR)
	cd $(GFORTH_DIR) && ./configure --prefix="$$PWD/_install" && $(MAKE) -j$$(nproc)

$(GFORTH_DIR):
	$(MAKE) setup-gforth

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

# Appbench: M. Anton Ertl's application benchmark suite (untracked shared tree).
APPBENCH_URL = https://www.complang.tuwien.ac.at/forth/appbench.zip
APPBENCH_DIR = appbench/appbench-1.4

.PHONY: setup-appbench
setup-appbench: $(APPBENCH_DIR)

$(APPBENCH_DIR):
	mkdir -p appbench
	wget -N -P appbench $(APPBENCH_URL)
	cd appbench && unzip -o appbench.zip
	@echo "appbench ready in $(APPBENCH_DIR)"

# Cold functional + performance grid across gforth / gforth-fast / rpyforth.
.PHONY: bench-appbench
bench-appbench: build-jit-stkfrag build-gforth setup-appbench $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_appbench.py func \
    	--iterations 5 --chart appbench.pdf

# Warm steady-state + per-iteration warm-up curve visualization
.PHONY: bench-appbench-curve
bench-appbench-curve: build-jit-stkfrag build-gforth setup-appbench $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_appbench.py steady \
    	--iterations 50 --pin 3 --pdf appbench-curve.pdf

# VFXForth: local install from bundled tree (no sudo, no /usr/local).
VFXFORTH_DIR     = vfxforth/VfxForth64Lin
VFXFORTH_INSTALL = vfxforth/_install
VFXFORTH_BINDIR  = $(VFXFORTH_INSTALL)/bin
VFXFORTH_BIN     = $(VFXFORTH_BINDIR)/VfxForth_x64_lin.elf
VFXFORTH         = $(VFXFORTH_BINDIR)/vfxforth
VFXFORTH_URL     = https://vfxforth.com/downloads/VfxCommunity/VfxForth_x64_lin.tar.gz
VFXFORTH_TARBALL = vfxforth/VfxForth_x64_lin.tar.gz

$(VFXFORTH_TARBALL):
	mkdir -p vfxforth
	wget -N -P vfxforth $(VFXFORTH_URL)

$(VFXFORTH_DIR)/.extracted: $(VFXFORTH_TARBALL)
	mkdir -p $(VFXFORTH_DIR)
	tar -zxf $< -C vfxforth
	@touch $@

$(VFXFORTH_BINDIR):
	mkdir -p $@

$(VFXFORTH_BIN): $(VFXFORTH_DIR)/Bin/VfxForth_x64_lin.elf | $(VFXFORTH_BINDIR) $(VFXFORTH_DIR)/.extracted
	cp -p $< $@

$(VFXFORTH_BINDIR)/VfxForthK_x64_lin.elf: $(VFXFORTH_DIR)/Bin/VfxForthK_x64_lin.elf | $(VFXFORTH_BINDIR) $(VFXFORTH_DIR)/.extracted
	cp -p $< $@

$(VFXFORTH_BINDIR)/x64.elf: $(VFXFORTH_DIR)/Bin/x64.elf | $(VFXFORTH_BINDIR) $(VFXFORTH_DIR)/.extracted
	cp -p $< $@

$(VFXFORTH_BINDIR)/stublin64.elf: $(VFXFORTH_DIR)/Bin/stublin64.elf | $(VFXFORTH_BINDIR) $(VFXFORTH_DIR)/.extracted
	cp -p $< $@

$(VFXFORTH_BINDIR)/libmpeparser64.so.0: $(VFXFORTH_DIR)/Bin/libmpeparser64.so.0 | $(VFXFORTH_BINDIR) $(VFXFORTH_DIR)/.extracted
	cp -p $< $@

$(VFXFORTH_BINDIR)/vfxsupp64.so.1: $(VFXFORTH_DIR)/Bin/vfxsupp64.so.1 | $(VFXFORTH_BINDIR) $(VFXFORTH_DIR)/.extracted
	cp -p $< $@

$(VFXFORTH): $(VFXFORTH_BIN) \
		$(VFXFORTH_BINDIR)/VfxForthK_x64_lin.elf \
		$(VFXFORTH_BINDIR)/x64.elf \
		$(VFXFORTH_BINDIR)/stublin64.elf \
		$(VFXFORTH_BINDIR)/libmpeparser64.so.0 \
		$(VFXFORTH_BINDIR)/vfxsupp64.so.1
	@printf '#!/bin/sh\nVFXROOT=$$(cd "$$(dirname "$$0")/.." && pwd)\nexport LD_LIBRARY_PATH="$$VFXROOT/bin$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}"\nexec "$$VFXROOT/bin/VfxForth_x64_lin.elf" "$$@"\n' > $@
	@chmod +x $@

.PHONY: download-vfxforth
download-vfxforth: $(VFXFORTH_TARBALL)

.PHONY: extract-vfxforth
extract-vfxforth: $(VFXFORTH_DIR)/.extracted

.PHONY: setup-vfxforth
setup-vfxforth: $(VFXFORTH)
	@echo "VFXForth installed locally in $(VFXFORTH_INSTALL)"
	@echo "Run with: $(VFXFORTH)"

.PHONY: test-vfxforth
test-vfxforth: setup-vfxforth
	@SMOKE=$(VFXFORTH_INSTALL)/_smoke_test.fs; \
	trap 'rm -f $$SMOKE' EXIT; \
	{ \
		echo '.( VFXForth local install smoke test started ) cr'; \
		echo '2 3 4 + * . cr'; \
		echo '.( VFXForth local install smoke test: OK ) cr'; \
		echo 'bye'; \
	} > $$SMOKE; \
	$(VFXFORTH) "include $$SMOKE"

.PHONY: uninstall-vfxforth
uninstall-vfxforth:
	@echo "Removing globally installed VFXForth files..."
	sudo rm -f /usr/local/bin/VfxForth*_x64_lin.elf
	sudo rm -f /usr/local/bin/x64.elf
	sudo rm -f /usr/local/bin/vfxlin64
	sudo rm -f /usr/local/bin/vfx64
	sudo rm -f /usr/lib/libmpeparser64.so.0
	sudo rm -f /usr/lib/vfxsupp64.so.1
	sudo rm -f /usr/lib/vfxsupp64.so.1.0.1
	rm -f $(HOME)/.VfxForth.ini
	rm -rf $(HOME)/.VfxForth
	@echo "VFXForth uninstalled."

# SwiftForth: local install from downloaded tarball (no sudo, no /usr/local).
SWIFTFORTH_URL     = https://dl.forth.com/downloads/SwiftForth-linux-eval.tgz
SWIFTFORTH_TARBALL = swiftforth/SwiftForth-linux-eval.tgz
SWIFTFORTH_DIR     = swiftforth/SwiftForth
SWIFTFORTH_INSTALL = swiftforth/_install
SWIFTFORTH_BIN     = $(SWIFTFORTH_INSTALL)/bin/sf

$(SWIFTFORTH_TARBALL):
	mkdir -p swiftforth
	wget -nc -P swiftforth $(SWIFTFORTH_URL)

$(SWIFTFORTH_DIR)/.extracted: $(SWIFTFORTH_TARBALL)
	mkdir -p $(SWIFTFORTH_DIR)
	tar -zxf $< -C swiftforth
	@touch $@

$(SWIFTFORTH_INSTALL)/bin:
	mkdir -p $@

$(SWIFTFORTH_BIN): $(SWIFTFORTH_DIR)/.extracted | $(SWIFTFORTH_INSTALL)/bin
	@printf '#!/bin/sh\nSFROOT=$$(cd "$$(dirname "$$0")/../.." && pwd)\ncd "$$SFROOT/SwiftForth" || exit 1\nexec "$$SFROOT/SwiftForth/bin/linux/sf64" "$$@"\n' > $@
	@chmod +x $@

.PHONY: download-swiftforth
download-swiftforth: $(SWIFTFORTH_TARBALL)

.PHONY: extract-swiftforth
extract-swiftforth: $(SWIFTFORTH_DIR)/.extracted

.PHONY: setup-swiftforth
setup-swiftforth: $(SWIFTFORTH_BIN)
	@echo "SwiftForth installed locally in $(SWIFTFORTH_INSTALL)"
	@echo "Run with: $(SWIFTFORTH_BIN)"

.PHONY: test-swiftforth
test-swiftforth: setup-swiftforth
	@$(SWIFTFORTH_BIN) 'cr .( SwiftForth local install smoke test started ) cr 2 3 4 + * . cr cr .( SwiftForth local install smoke test: OK ) cr bye'

# Ablation analysis
ABLATION_RESULTS ?= logs/analysis/7038abb-dirty/results.json
ABLATION_STEADY ?= logs/analysis/warm_steady.json

.PHONY: bench-ablation-render
bench-ablation-render: $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_ablation.py render \
    	$(ABLATION_RESULTS) --steady-json $(ABLATION_STEADY)
