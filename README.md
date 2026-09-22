# Crop yield and weather in the Republic of Moldova

Predicting raion-level crop yields from weather, and testing how much the
choice of cross-validation scheme changes the answer.

Course project for Știința Datelor. The pipeline is also the modelling chapter
of a master's thesis on climate adaptation in Moldovan agriculture.

## The question

Two questions, in order of importance:

1. Does gradient boosting predict crop yields better than a parametric response
   function with raion fixed effects?
2. Does the answer to question 1 survive an honest validation scheme?

On a raion-by-year panel, neighbouring raions live through the same drought. A
random train/test split therefore puts observations from 2012 on both sides of
the split, and rewards a model for having already seen that year. Blocking by
year removes the leak.

## Main result

Under random 5-fold validation gradient boosting wins comfortably. Under
leave-one-year-out the ranking reverses for every major crop.

RMSE on log yield:

| crop | model | random 5-fold | leave-one-year-out | leave-one-raion-out |
|---|---|---|---|---|
| griu | baseline FE | 0.312 | **0.343** | 0.349 |
| griu | gradient boosting | **0.191** | 0.385 | 0.224 |
| porumb | baseline FE | 0.576 | **0.665** | 0.596 |
| porumb | gradient boosting | **0.370** | 0.766 | 0.412 |
| orz | baseline FE | 0.345 | **0.376** | 0.398 |
| orz | gradient boosting | **0.258** | 0.470 | 0.309 |
| floarea_soarelui | baseline FE | 0.259 | **0.287** | 0.291 |
| floarea_soarelui | gradient boosting | **0.186** | 0.343 | 0.217 |

Moving from random splitting to year blocks costs the boosting model 82 to 107
percent of its RMSE, against 9 to 15 percent for the parametric baseline. Almost
all of the apparent advantage of the flexible model was leakage.

Constraining the boosting model to be monotone in heat changes accuracy by less
than half a percent either way, so the leak is about unseen years, not about
functional form. The constraint buys a readable response curve for free.

Leaving out a whole raion is much gentler than leaving out a whole year. Places
are more exchangeable than years: a model that has seen 32 raions can guess the
33rd, but a model that has never seen a drought cannot guess one.

## The agronomy checks out

Killing degree days are cumulative degrees above 30 C over April to August. The
parametric model gives the effect of one unit on yield:

| crop | effect per KDD unit | std. error | n |
|---|---|---|---|
| porumb | -4.07% | 0.29 | 626 |
| soia | -3.19% | 0.48 | 387 |
| floarea_soarelui | -1.64% | 0.13 | 627 |
| cartofi | -1.57% | 0.43 | 519 |
| sfecla | -1.23% | 0.31 | 254 |
| legume | -1.06% | 0.34 | 595 |
| griu | -0.63% | 0.16 | 627 |
| orz | -0.55% | 0.17 | 627 |

The ordering is the phenological one. Winter cereals fill their grain before the
July heat peak and lose least; summer crops stand exposed and lose most. Nothing
in the model was told this, which is the point: it is a validity check, not a
result.

`results/figures/response_kdd.png` shows the same thing recovered by gradient
boosting under monotonic constraints, averaged over twelve seeds and cut at the
90th percentile of heat exposure. Wheat and barley flatten out around three
percent; maize reaches eighteen and soya twenty-eight.

## Data

Both sources are open, need no registration and no API key, and are downloaded
by the scripts. Nothing is collected by hand.

- **Yields.** National Bureau of Statistics of Moldova, PxWeb API, table
  `AGR020600reg`: sown area, average yield and gross output by crop and raion.
  Agricultural enterprises and peasant farms.
- **Weather.** ERA5 reanalysis served by Open-Meteo, daily maximum and minimum
  temperature and precipitation, for each raion centre.

Resulting panel: 4262 observations, 33 raions, 8 crops, 2007 to 2025.

## Running it

```
pip install -r requirements.txt
make all
```

`make all` fetches the data, builds the panel, produces the figures and runs the
validation experiment. Raw downloads are cached in `data/raw`, so re-runs are
fast; delete a file there to force a refresh. The full run takes a few minutes,
most of it in the leave-one-raion-out loop.

Individual stages: `make data`, `make panel`, `make explore`, `make model`,
`make validate`.

## Layout

```
config.yaml              every parameter the analysis depends on
src/01_fetch_yields.py   BNS PxWeb API -> data/raw/yields.json
src/02_fetch_weather.py  geocode raions, ERA5 -> data/raw/weather.json
src/03_build_panel.py    degree days and seasonal features -> panel.csv
src/04_explore.py        summary table, yield series, shock correlations
src/05_model_baseline.py parametric response function, coefficients
src/06_model_ml.py       gradient boosting, monotonic response curve
src/07_validation.py     three models, three validation schemes
src/models.py            both models behind one fit/predict interface
src/utils.py             paths, config, logging
```

Features are built once in `03` and consumed everywhere else, so the two models
see exactly the same inputs.

## Known limitations

- **The response curve is still a staircase.** Monotonic constraints, seed
  averaging and cutting the sparse tail make it readable, but partial dependence
  from a tree ensemble is a step function and sugar beet, with 254 observations,
  still jumps. Read the level, not the exact breakpoint.
- **Weather is taken at the raion centre**, not averaged over the sown area.
- **Household plots are outside the BNS table.** Results describe agricultural
  enterprises and peasant farms.
- **Inputs are not observed.** Fertiliser, seed variety and prices are absorbed
  into the raion effect and the trend, so the weather coefficients are net of
  whatever adaptation already happened.
- **Irrigation is not controlled for** and would bias the precipitation term
  wherever it is used.

## Next steps

1. Weighted weather aggregation using sown area.
2. Nested blocking (year and raion together) as a stricter test.
3. A spline or generalised additive learner, to compare against the staircase.
4. Feed the estimated response and the shock covariance into a crop-mix
   optimiser. That is the thesis, not this project.
