import { FormEvent, useState } from "react";
import { defaultRiskRequest } from "../api";
import { RiskRequest } from "../types";

interface AssessmentInputFormProps {
  submitting: boolean;
  onSubmit: (payload: RiskRequest) => Promise<void>;
}

const heatwavePreset: RiskRequest = {
  ...defaultRiskRequest,
  weather: {
    temperature_c: 44,
    humidity_percent: 42,
    heatwave_alert: true,
    rainfall_mm: 4
  },
  power_grid: {
    current_load_percent: 94,
    peak_load_percent: 99,
    historical_failure_rate_percent: 20,
    transformer_age_years: 29
  },
  water_system: {
    pressure_variance_percent: 33,
    pipeline_age_years: 30,
    recent_repairs_last_30_days: 4
  },
  traffic_system: {
    congestion_index: 84,
    signal_failure_reports_last_7_days: 8
  },
  population_density_per_km2: 14500
};

const highDensityPreset: RiskRequest = {
  ...defaultRiskRequest,
  city: "Bayhaven",
  weather: {
    temperature_c: 33,
    humidity_percent: 58,
    heatwave_alert: false,
    rainfall_mm: 18
  },
  power_grid: {
    current_load_percent: 82,
    peak_load_percent: 90,
    historical_failure_rate_percent: 14,
    transformer_age_years: 22
  },
  water_system: {
    pressure_variance_percent: 28,
    pipeline_age_years: 27,
    recent_repairs_last_30_days: 3
  },
  traffic_system: {
    congestion_index: 91,
    signal_failure_reports_last_7_days: 11
  },
  population_density_per_km2: 17600
};

function inputClassName(): string {
  return "lux-input";
}

function sectionTitleClassName(): string {
  return "text-[11px] font-semibold uppercase tracking-[0.2em] text-gridmind-gold/85";
}

function fieldsetClassName(): string {
  return "rounded-xl border border-gridmind-gold/20 bg-gridmind-midnight/35 p-3";
}

function labelClassName(): string {
  return "text-sm text-gridmind-haze";
}

function presetButtonClassName(): string {
  return "rounded-lg border border-gridmind-gold/30 bg-gridmind-carbon/70 px-3 py-1.5 text-xs font-semibold text-gridmind-pearl transition hover:border-gridmind-gold/60 hover:bg-gridmind-carbon";
}

export function AssessmentInputForm({ submitting, onSubmit }: AssessmentInputFormProps) {
  const [form, setForm] = useState<RiskRequest>({
    ...defaultRiskRequest,
    timestamp: new Date().toISOString()
  });

  function setTimestampNow(payload: RiskRequest): RiskRequest {
    return { ...payload, timestamp: new Date().toISOString() };
  }

  function updateWeather<K extends keyof RiskRequest["weather"]>(key: K, value: RiskRequest["weather"][K]) {
    setForm((prev) => ({ ...prev, weather: { ...prev.weather, [key]: value } }));
  }

  function updatePower<K extends keyof RiskRequest["power_grid"]>(key: K, value: RiskRequest["power_grid"][K]) {
    setForm((prev) => ({ ...prev, power_grid: { ...prev.power_grid, [key]: value } }));
  }

  function updateWater<K extends keyof RiskRequest["water_system"]>(key: K, value: RiskRequest["water_system"][K]) {
    setForm((prev) => ({ ...prev, water_system: { ...prev.water_system, [key]: value } }));
  }

  function updateTraffic<K extends keyof RiskRequest["traffic_system"]>(
    key: K,
    value: RiskRequest["traffic_system"][K]
  ) {
    setForm((prev) => ({ ...prev, traffic_system: { ...prev.traffic_system, [key]: value } }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload = setTimestampNow(form);
    setForm(payload);
    await onSubmit(payload);
  }

  function loadPreset(preset: RiskRequest) {
    setForm(setTimestampNow(preset));
  }

  return (
    <section className="panel anim-sheen">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-heading text-2xl text-gridmind-pearl">Executive Scenario Console</h2>
          <p className="text-xs uppercase tracking-[0.16em] text-gridmind-gold/80">
            Precision input surface for predictive assessment
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className={presetButtonClassName()}
            onClick={() => loadPreset(defaultRiskRequest)}
            disabled={submitting}
          >
            Baseline Preset
          </button>
          <button
            type="button"
            className={presetButtonClassName()}
            onClick={() => loadPreset(heatwavePreset)}
            disabled={submitting}
          >
            Heatwave Preset
          </button>
          <button
            type="button"
            className={presetButtonClassName()}
            onClick={() => loadPreset(highDensityPreset)}
            disabled={submitting}
          >
            High Density Preset
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
          <label className={labelClassName()}>
            City
            <input
              className={inputClassName()}
              value={form.city}
              onChange={(event) => setForm((prev) => ({ ...prev, city: event.target.value }))}
              required
            />
          </label>
          <label className={labelClassName()}>
            Population Density / km2
            <input
              className={inputClassName()}
              type="number"
              min={0}
              max={150000}
              value={form.population_density_per_km2}
              onChange={(event) =>
                setForm((prev) => ({ ...prev, population_density_per_km2: Number(event.target.value) }))
              }
              required
            />
          </label>
          <label className={labelClassName()}>
            Heatwave Alert
            <select
              className={inputClassName()}
              value={form.weather.heatwave_alert ? "true" : "false"}
              onChange={(event) => updateWeather("heatwave_alert", event.target.value === "true")}
            >
              <option value="false">False</option>
              <option value="true">True</option>
            </select>
          </label>
          <label className={labelClassName()}>
            Timestamp (Auto)
            <input className={inputClassName()} value={new Date(form.timestamp).toLocaleString()} disabled />
          </label>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <fieldset className={fieldsetClassName()}>
            <legend className={sectionTitleClassName()}>Weather</legend>
            <div className="mt-2 grid gap-3 md:grid-cols-2">
              <label className={labelClassName()}>
                Temperature (C)
                <input
                  className={inputClassName()}
                  type="number"
                  min={-30}
                  max={65}
                  step="0.1"
                  value={form.weather.temperature_c}
                  onChange={(event) => updateWeather("temperature_c", Number(event.target.value))}
                  required
                />
              </label>
              <label className={labelClassName()}>
                Humidity (%)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.weather.humidity_percent}
                  onChange={(event) => updateWeather("humidity_percent", Number(event.target.value))}
                  required
                />
              </label>
              <label className={`${labelClassName()} md:col-span-2`}>
                Rainfall (mm)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={600}
                  step="0.1"
                  value={form.weather.rainfall_mm}
                  onChange={(event) => updateWeather("rainfall_mm", Number(event.target.value))}
                  required
                />
              </label>
            </div>
          </fieldset>

          <fieldset className={fieldsetClassName()}>
            <legend className={sectionTitleClassName()}>Power Grid</legend>
            <div className="mt-2 grid gap-3 md:grid-cols-2">
              <label className={labelClassName()}>
                Current Load (%)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.power_grid.current_load_percent}
                  onChange={(event) => updatePower("current_load_percent", Number(event.target.value))}
                  required
                />
              </label>
              <label className={labelClassName()}>
                Peak Load (%)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.power_grid.peak_load_percent}
                  onChange={(event) => updatePower("peak_load_percent", Number(event.target.value))}
                  required
                />
              </label>
              <label className={labelClassName()}>
                Historical Failure Rate (%)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.power_grid.historical_failure_rate_percent}
                  onChange={(event) => updatePower("historical_failure_rate_percent", Number(event.target.value))}
                  required
                />
              </label>
              <label className={labelClassName()}>
                Transformer Age (years)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={80}
                  step="0.1"
                  value={form.power_grid.transformer_age_years}
                  onChange={(event) => updatePower("transformer_age_years", Number(event.target.value))}
                  required
                />
              </label>
            </div>
          </fieldset>

          <fieldset className={fieldsetClassName()}>
            <legend className={sectionTitleClassName()}>Water System</legend>
            <div className="mt-2 grid gap-3 md:grid-cols-2">
              <label className={labelClassName()}>
                Pressure Variance (%)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.water_system.pressure_variance_percent}
                  onChange={(event) => updateWater("pressure_variance_percent", Number(event.target.value))}
                  required
                />
              </label>
              <label className={labelClassName()}>
                Pipeline Age (years)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.water_system.pipeline_age_years}
                  onChange={(event) => updateWater("pipeline_age_years", Number(event.target.value))}
                  required
                />
              </label>
              <label className={`${labelClassName()} md:col-span-2`}>
                Recent Repairs (last 30 days)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={50}
                  value={form.water_system.recent_repairs_last_30_days}
                  onChange={(event) => updateWater("recent_repairs_last_30_days", Number(event.target.value))}
                  required
                />
              </label>
            </div>
          </fieldset>

          <fieldset className={fieldsetClassName()}>
            <legend className={sectionTitleClassName()}>Traffic System</legend>
            <div className="mt-2 grid gap-3 md:grid-cols-2">
              <label className={labelClassName()}>
                Congestion Index
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  step="0.1"
                  value={form.traffic_system.congestion_index}
                  onChange={(event) => updateTraffic("congestion_index", Number(event.target.value))}
                  required
                />
              </label>
              <label className={labelClassName()}>
                Signal Failure Reports (7d)
                <input
                  className={inputClassName()}
                  type="number"
                  min={0}
                  max={100}
                  value={form.traffic_system.signal_failure_reports_last_7_days}
                  onChange={(event) =>
                    updateTraffic("signal_failure_reports_last_7_days", Number(event.target.value))
                  }
                  required
                />
              </label>
            </div>
          </fieldset>
        </div>

        <div className="flex items-center justify-end">
          <button
            type="submit"
            disabled={submitting}
            className="rounded-lg border border-gridmind-gold/50 bg-gradient-to-r from-gridmind-gold/80 to-gridmind-goldDeep px-5 py-2 text-sm font-semibold text-gridmind-obsidian transition hover:shadow-glow disabled:cursor-not-allowed disabled:opacity-60"
          >
            {submitting ? "Running..." : "Run Assessment"}
          </button>
        </div>
      </form>
    </section>
  );
}

