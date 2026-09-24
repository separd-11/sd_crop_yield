PY ?= python

.PHONY: all data panel explore model validate mechanism uncertainty baselines app clean

all: explore mechanism uncertainty baselines

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

mechanism: validate
	$(PY) src/08_extrapolation.py
	$(PY) src/09_shrinkage.py
	$(PY) src/10_map.py

uncertainty: validate
	$(PY) src/11_uncertainty.py

baselines: panel
	$(PY) src/12_naive_baselines.py
	$(PY) src/13_rolling_origin.py
	$(PY) src/14_model_zoo.py
	$(PY) src/15_pooled.py

app:
	streamlit run app/app.py

clean:
	rm -f data/processed/*.csv results/tables/*.csv results/figures/*.png
