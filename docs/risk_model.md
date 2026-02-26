# GridMind AI Risk Model (Hybrid)

## 1) Model Structure

Each system score is computed in five steps:

1. Normalize key signals to `0-100`.
2. Compute weighted base score.
3. Add rule-threshold bonuses.
4. Apply compound amplification (`+10%` or `+15%` when 2+ factors intersect).
5. Clamp to `0-100` and classify:
   - `LOW: 0-35`
   - `MODERATE: 36-65`
   - `HIGH: 66-85`
   - `CRITICAL: 86-100`

## 2) System Formulas

### Power Grid

Normalized terms:

- `CL = current_load_percent`
- `PL = peak_load_percent`
- `TA = min((transformer_age_years / 40) * 100, 100)`
- `HF = min(historical_failure_rate_percent * 4, 100)`
- `HW = 100 if heatwave_alert else 0`

Base:

`Power_base = 0.32*CL + 0.22*PL + 0.18*TA + 0.18*HF + 0.10*HW`

Threshold bonuses:

- `+8` if `CL > 85`
- `+7` if `PL > 90`
- `+6` if `transformer_age_years > 20`
- `+5` if `heatwave_alert = true`
- `+6` if `historical_failure_rate_percent > 15`

Compound:

- `x1.10` when exactly 2 high factors intersect
- `x1.15` when 3+ high factors intersect
- Extra `x1.08` if `(heatwave_alert = true AND CL > 85)`

### Transformer Overload

`Transformer_base = 0.38*PL + 0.34*CL + 0.20*TA35 + 0.08*HF35`

Where:
- `TA35 = min((transformer_age_years / 35)*100, 100)`
- `HF35 = min(historical_failure_rate_percent * 3.5, 100)`

Bonuses:
- `+10` if `PL > 90`
- `+8` if `CL > 85`
- `+9` if `transformer_age_years > 20`
- `+4` if `historical_failure_rate_percent > 15`
- `+4` if `heatwave_alert = true`

Compound:
- `x1.10` or `x1.15` by high-factor count.

### Water Pipeline

`Water_base = 0.36*PV + 0.28*PA + 0.22*RN + 0.14*RR`

Where:
- `PV = pressure_variance_percent`
- `PA = min((pipeline_age_years / 60)*100, 100)`
- `RN = min((rainfall_mm / 80)*100, 100)`
- `RR = min((recent_repairs_last_30_days / 10)*100, 100)`

Bonuses:
- `+10` if `PV > 30`
- `+9` if `pipeline_age_years > 25`
- `+8` if `rainfall_mm > 40`
- `+7` if `recent_repairs_last_30_days > 3`

Compound:
- `x1.10` or `x1.15`.

### Traffic Infrastructure

`Traffic_base = 0.48*CI + 0.30*SF + 0.22*PD`

Where:
- `CI = congestion_index`
- `SF = min((signal_failure_reports_last_7_days / 15)*100, 100)`
- `PD = min((population_density_per_km2 / 20000)*100, 100)`

Bonuses:
- `+10` if `CI > 75`
- `+8` if `signal_failure_reports_last_7_days > 5`
- `+7` if `population_density_per_km2 > 10000`

Compound:
- `x1.10` or `x1.15`.

## 3) Cascading Failure Logic

After individual system scoring:

- If `power_grid_score >= 66`:
  - traffic risk increases by `impact = min(14, 5 + 0.12*(power_score - 66))`
  - overall multiplier `+0.03`
- If `water_score >= 66` and density `>10000`:
  - traffic risk increases by `impact = min(12, 4 + 0.10*(water_score - 66))`
  - overall multiplier `+0.03`
- If congestion `>75`:
  - add emergency response delay cascade flag
  - overall multiplier `+0.02`
- If high density and aging infrastructure intersect (`density>10000` and transformer age `>20` or pipeline age `>25`):
  - overall multiplier `+0.05`

## 4) Overall Risk Score

Weighted sum:

`Overall_base = 0.33*Power + 0.17*Transformer + 0.25*Water + 0.25*Traffic`

Final:

`Overall_score = clamp(Overall_base * cascade_multiplier, 0, 100)`

## 5) 24h and 72h Outlook

24h:
- Template derived from overall class + top two highest-risk systems.

72h projected delta:
- `+8` heatwave and high load
- `+4` heatwave only
- `+3` peak load > 90
- `+5` rainfall > 40
- `+2` pressure variance > 30
- `+3` congestion > 75
- `+2` signal failures > 5
- `+2` repairs > 3
- `+3` high-density aging intersection

`Projected_72h = clamp(Overall_score + delta, 0, 100)`

## 6) Confidence Scoring

Start `95`, then deduct:

- `-5` for each missing input field
- `-10` if historical failure data is low (`<5`) or missing
- `-8` if no trend indicators are present (no heatwave, low rainfall/load/congestion/repairs signals)

`Confidence = clamp(result, 0, 100)`

## 7) Weight Justification

- Power and traffic dominate immediate public-impact failures, hence highest combined weight (`58%` overall).
- Water remains high-impact but slower than power failure onset, weighted `25%`.
- Transformer overload is represented separately to expose asset-level electrical fragility while avoiding double-count dominance (`17%`).
- Cascade multiplier captures cross-system fragility not visible in isolated subsystem scoring.

## 8) Supervised ML Overlay (Optional)

When a trained bundle is present (`backend/models/gridmind_ml_bundle_numpy.json`), the API overlays rule-based risk scores with model probabilities:

- `system_score_final = (1 - blend_weight) * rule_score + blend_weight * (ml_probability * 100)`
- `overall_score_final = (1 - blend_weight) * rule_overall + blend_weight * (overall_probability * 100)`

`blend_weight` is learned from holdout quality (`recommended_ml_blend_weight` in training report).

This keeps explainability from rules while improving calibration from historical labels.
