PY ?= python

.PHONY: all data panel explore model validate clean

all: validate

data:
	$(PY) src/01_fetch_yields.py
	$(PY) src/02_fetch_weather.py

panel: data
	$(PY) src/03_build_panel.py

explore: panel
	$(PY) src/04_explore.py

model: panel
	$(PY) src/05_model_baseline.py
	$(PY) src/06_model_ml.py

validate: model
	$(PY) src/07_validation.py

clean:
	rm -f data/processed/*.csv results/tables/*.csv results/figures/*.png
