PYTHON2 = ./_pypy_binary/bin/python2
RPYTHON = ./pypy/rpython/bin/rpython
RPYTHON_ARGS =
TARGET = targetrpyforth

VENV = .venv
PLOT_PY = $(VENV)/bin/python

GFORTH_DIR = gforth-0.7.9
GFORTH_FAST = ./$(GFORTH_DIR)/gforth-fast
GFORTH = ./$(GFORTH_DIR)/gforth

VFXFORTH = ./vfxforth.sh
SWIFTFORTH = ./swiftforth.sh

# ---------------------------------------------------------------------------
# RPyForth builds
# ---------------------------------------------------------------------------

.PHONY: build
build: build-jit build-jit-stkfrag build-jit-novirt

.PHONY: build-all
build-all: build

.PHONY: build-interp
build-interp: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. $(PYTHON2) $(RPYTHON) -O2 $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit
build-jit: _pypy_binary/bin/python setup-pypy
	RPYFORTH_VIRTUALIZE=1 PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag
build-jit-stkfrag: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_LAYOUT=fragment PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag-floatfrag
build-jit-stkfrag-floatfrag: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_LAYOUT=fragment-float PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag-floatfrag $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag-frameonly
build-jit-stkfrag-frameonly: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_LAYOUT=frame-only PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag-frameonly $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag-ntop4
build-jit-stkfrag-ntop4: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_LAYOUT=ntop4 PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag-ntop4 $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag-ntop8
build-jit-stkfrag-ntop8: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_LAYOUT=ntop8 PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag-ntop8 $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-stkfrag-ntop16
build-jit-stkfrag-ntop16: _pypy_binary/bin/python setup-pypy
	RPYFORTH_STACK_LAYOUT=ntop16 PYTHONPATH=. RPYFORTH_EXE_NAME=rpyforth-c-stkfrag-ntop16 $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

.PHONY: build-jit-novirt
build-jit-novirt: _pypy_binary/bin/python setup-pypy
	RPYFORTH_EXE_NAME=rpyforth-c-novirt PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyforth/$(TARGET).py

# ---------------------------------------------------------------------------
# RPyFactor / Joy builds (build only; no test/bench Makefile entries)
# ---------------------------------------------------------------------------

.PHONY: build-factor-all
build-factor-all: build-factor build-factor-stkfrag build-factor-stkfrag-vable

.PHONY: build-factor
build-factor: _pypy_binary/bin/python setup-pypy
	RPYFACTOR_EXE_NAME=rpyfactor-c PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyfactor/targetrpyfactor.py

.PHONY: build-factor-stkfrag
build-factor-stkfrag: _pypy_binary/bin/python setup-pypy
	RPYFACTOR_STACK_FRAGMENT=1 RPYFACTOR_EXE_NAME=rpyfactor-c-stkfrag PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyfactor/targetrpyfactor.py

.PHONY: build-factor-stkfrag-vable
build-factor-stkfrag-vable: _pypy_binary/bin/python setup-pypy
	RPYFACTOR_STACK_FRAGMENT=1 RPYFACTOR_STACK_VABLE=1 RPYFACTOR_EXE_NAME=rpyfactor-c-stkfrag-vable PYTHONPATH=. $(PYTHON2) $(RPYTHON) -Ojit $(RPYTHON_ARGS) rpyfactor/targetrpyfactor.py

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

.PHONY: test
test: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyforth/test -vv -s

.PHONY: test-factor
test-factor: _pypy_binary/bin/python setup-pypy
	PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyfactor/test -q

.PHONY: coverage
coverage:
	python3 check_coverage.py

# ---------------------------------------------------------------------------
# Toolchain / baselines
# ---------------------------------------------------------------------------

.PHONY: setup-pypy
setup-pypy:
	if [ ! -d pypy ]; then git clone https://github.com/pypy/pypy.git --depth=1; fi

_pypy_binary/bin/python:
	mkdir -p _pypy_binary
	python3 get_pypy_to_download.py
	tar -C _pypy_binary --strip-components=1 -xf pypy.tar.bz2
	rm pypy.tar.bz2
	./_pypy_binary/bin/python -m ensurepip
	./_pypy_binary/bin/python -mpip install "hypothesis<4.40" junit_xml coverage==5.5 "pdbpp==0.10.3"

$(PLOT_PY):
	python3 -m venv $(VENV)
	$(PLOT_PY) -m pip install --upgrade pip
	$(PLOT_PY) -m pip install matplotlib

.PHONY: setup-plot
setup-plot: $(PLOT_PY)

# gforth
.PHONY: setup-gforth
setup-gforth: $(GFORTH_DIR)/gforth-fast

$(GFORTH_DIR)/gforth-fast: | $(GFORTH_DIR)
	cd $(GFORTH_DIR) && ./configure --prefix="$$PWD/_install" && $(MAKE) -j$$(nproc)

$(GFORTH_DIR):
	wget -N https://www.complang.tuwien.ac.at/forth/gforth/Snapshots/current/gforth.tar.xz
	tar xJf gforth.tar.xz
	src=$$(tar tJf gforth.tar.xz | sed 's#/.*##' | grep -m1 '^gforth-'); \
	dest=$$(printf '%s' "$$src" | sed -E 's/_[0-9]{8}$$//'); \
	if [ "$$src" != "$$dest" ]; then rm -rf "$$dest"; mv "$$src" "$$dest"; fi

# Factor (rpyfactor external baseline)
FACTOR_VERSION = 0.101
FACTOR_TARBALL = factor-linux-x86-64-$(FACTOR_VERSION).tar.gz
FACTOR_URL = https://downloads.factorcode.org/releases/$(FACTOR_VERSION)/$(FACTOR_TARBALL)
FACTOR_DIR = factor
FACTOR = ./$(FACTOR_DIR)/factor

.PHONY: setup-factor
setup-factor: $(FACTOR)

$(FACTOR):
	wget -N $(FACTOR_URL)
	rm -rf $(FACTOR_DIR).tmp
	mkdir -p $(FACTOR_DIR).tmp
	tar -C $(FACTOR_DIR).tmp --strip-components=1 -xzf $(FACTOR_TARBALL)
	rm -rf $(FACTOR_DIR)
	mv $(FACTOR_DIR).tmp $(FACTOR_DIR)
	@test -x $(FACTOR)
	@$(FACTOR) -e='"rpyforth setup-factor ok" print' || \
	  (echo "Factor binary present but smoke failed"; exit 1)
	@echo "Factor ready at $(FACTOR)"

# VFXForth (local, no sudo). Repo wrapper ./vfxforth.sh expects
# vfxforth/_install/bin/vfxforth.
VFXFORTH_DIR     = vfxforth/VfxForth64Lin
VFXFORTH_INSTALL = vfxforth/_install
VFXFORTH_BINDIR  = $(VFXFORTH_INSTALL)/bin
VFXFORTH_LOCAL   = $(VFXFORTH_BINDIR)/vfxforth
VFXFORTH_URL     = https://vfxforth.com/downloads/VfxCommunity/VfxForth_x64_lin.tar.gz
VFXFORTH_TARBALL = vfxforth/VfxForth_x64_lin.tar.gz

.PHONY: setup-vfxforth
setup-vfxforth: $(VFXFORTH_LOCAL)
	@echo "VFXForth ready: $(VFXFORTH) -> $(VFXFORTH_LOCAL)"

$(VFXFORTH_LOCAL):
	mkdir -p vfxforth
	@if [ ! -f $(VFXFORTH_TARBALL) ]; then \
	  wget -N -P vfxforth $(VFXFORTH_URL); \
	fi
	@if [ ! -f $(VFXFORTH_DIR)/.extracted ]; then \
	  mkdir -p $(VFXFORTH_DIR); \
	  tar -zxf $(VFXFORTH_TARBALL) -C vfxforth; \
	  touch $(VFXFORTH_DIR)/.extracted; \
	fi
	mkdir -p $(VFXFORTH_BINDIR)
	cp -p $(VFXFORTH_DIR)/Bin/VfxForth_x64_lin.elf $(VFXFORTH_BINDIR)/
	cp -p $(VFXFORTH_DIR)/Bin/VfxForthK_x64_lin.elf $(VFXFORTH_BINDIR)/
	cp -p $(VFXFORTH_DIR)/Bin/x64.elf $(VFXFORTH_BINDIR)/
	cp -p $(VFXFORTH_DIR)/Bin/stublin64.elf $(VFXFORTH_BINDIR)/
	cp -p $(VFXFORTH_DIR)/Bin/libmpeparser64.so.0 $(VFXFORTH_BINDIR)/
	cp -p $(VFXFORTH_DIR)/Bin/vfxsupp64.so.1 $(VFXFORTH_BINDIR)/
	@printf '#!/bin/sh\nVFXROOT=$$(cd "$$(dirname "$$0")/.." && pwd)\nexport LD_LIBRARY_PATH="$$VFXROOT/bin$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}"\nexec "$$VFXROOT/bin/VfxForth_x64_lin.elf" "$$@"\n' > $@
	@chmod +x $@

# SwiftForth (local, no sudo). Repo wrapper ./swiftforth.sh expects
# swiftforth/_install/bin/sf.
SWIFTFORTH_URL     = https://dl.forth.com/downloads/SwiftForth-linux-eval.tgz
SWIFTFORTH_TARBALL = swiftforth/SwiftForth-linux-eval.tgz
SWIFTFORTH_DIR     = swiftforth/SwiftForth
SWIFTFORTH_INSTALL = swiftforth/_install
SWIFTFORTH_LOCAL   = $(SWIFTFORTH_INSTALL)/bin/sf

.PHONY: setup-swiftforth
setup-swiftforth: $(SWIFTFORTH_LOCAL)
	@echo "SwiftForth ready: $(SWIFTFORTH) -> $(SWIFTFORTH_LOCAL)"

$(SWIFTFORTH_LOCAL):
	mkdir -p swiftforth $(SWIFTFORTH_INSTALL)/bin
	@if [ ! -f $(SWIFTFORTH_TARBALL) ]; then \
	  wget -nc -P swiftforth $(SWIFTFORTH_URL); \
	fi
	@if [ ! -f $(SWIFTFORTH_DIR)/.extracted ]; then \
	  mkdir -p $(SWIFTFORTH_DIR); \
	  tar -zxf $(SWIFTFORTH_TARBALL) -C swiftforth; \
	  touch $(SWIFTFORTH_DIR)/.extracted; \
	fi
	@printf '#!/bin/sh\nSFROOT=$$(cd "$$(dirname "$$0")/../.." && pwd)\ncd "$$SFROOT/SwiftForth" || exit 1\nexec "$$SFROOT/SwiftForth/bin/linux/sf64" "$$@"\n' > $@
	@chmod +x $@
# Appbench sources
APPBENCH_URL = https://www.complang.tuwien.ac.at/forth/appbench.zip
APPBENCH_DIR = appbench/appbench-1.4

.PHONY: setup-appbench
setup-appbench: $(APPBENCH_DIR)

$(APPBENCH_DIR):
	mkdir -p appbench
	wget -N -P appbench $(APPBENCH_URL)
	cd appbench && unzip -o appbench.zip
	@echo "appbench ready in $(APPBENCH_DIR)"

# All Forth baselines used by shootout / appbench
.PHONY: setup-baselines
setup-baselines: setup-gforth setup-vfxforth setup-swiftforth

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

RPYFORTH_STKFRAG = rpyforth-c-stkfrag

# Build the benchmark binary only when it does not exist yet; use
# `make build-jit-stkfrag` explicitly to force a re-translation.
$(RPYFORTH_STKFRAG):
	$(MAKE) build-jit-stkfrag

SHOOTOUT_COMPARE = \
	--compare ./rpyforth-c-stkfrag \
	--compare $(GFORTH_FAST) \
	--compare $(VFXFORTH) \
	--compare $(SWIFTFORTH)

BENCH_DEPS = $(RPYFORTH_STKFRAG) setup-baselines $(PLOT_PY)

.PHONY: bench-shootout
bench-shootout: $(BENCH_DEPS)
	@$(PLOT_PY) benchmark/run_shootout.py \
		$(SHOOTOUT_COMPARE) \
		--exclude curve/ --iterations 5 \
		--chart compare.pdf

.PHONY: bench-shootout-curve
bench-shootout-curve: $(BENCH_DEPS)
	@$(PLOT_PY) benchmark/run_shootout.py \
		$(SHOOTOUT_COMPARE) \
		--only curve/ --iterations 5 \
		--curve-chart warmup.pdf

.PHONY: bench-appbench
bench-appbench: $(BENCH_DEPS) setup-appbench
	@$(PLOT_PY) benchmark/run_appbench.py func \
		--engines rpyforth gforth-fast vfxforth swiftforth \
		--iterations 5 --chart appbench.pdf

.PHONY: bench-appbench-curve
bench-appbench-curve: $(BENCH_DEPS) setup-appbench
	@$(PLOT_PY) benchmark/run_appbench.py steady \
		--engines rpyforth gforth-fast vfxforth swiftforth \
		--iterations 50 --pin 3 --pdf appbench-curve.pdf

.PHONY: bench-factor
bench-factor: setup-plot
	@$(PLOT_PY) benchmark/factor/run_factor.py --iterations 3

.PHONY: sweep-framesize
sweep-framesize: $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_param_sweep.py \
		--iterations 3 --pin 2 \
		--pdf sweep-framesize.pdf

ABLATION_RESULTS ?= logs/analysis/7038abb-dirty/results.json
ABLATION_STEADY ?= logs/analysis/warm_steady.json

.PHONY: bench-ablation-render
bench-ablation-render: $(PLOT_PY)
	@$(PLOT_PY) benchmark/run_ablation.py render \
		$(ABLATION_RESULTS) --steady-json $(ABLATION_STEADY)
