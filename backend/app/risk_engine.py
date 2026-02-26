from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ai_reasoning import generate_executive_summary
from .models import (
    OverallRisk,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    SystemRisk,
    classify_risk,
)


@dataclass
class SystemComputation:
    score: float
    drivers: list[str]
    compound_flags: list[str]
    high_factor_count: int


def _cap(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _num(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def _age_norm(age: float, ceiling: float) -> float:
    return _cap((age / ceiling) * 100.0)


def _compound_multiplier(high_factor_count: int) -> float:
    if high_factor_count >= 3:
        return 1.15
    if high_factor_count == 2:
        return 1.10
    return 1.0


def _compound_percent(multiplier: float) -> int:
    return int(round((multiplier - 1.0) * 100.0))


def _build_missing_fields(payload: RiskAssessmentRequest) -> list[str]:
    fields: list[str] = []
    nested: dict[str, dict[str, Any]] = {
        "weather": payload.weather.model_dump(),
        "power_grid": payload.power_grid.model_dump(),
        "water_system": payload.water_system.model_dump(),
        "traffic_system": payload.traffic_system.model_dump(),
    }
    for parent, items in nested.items():
        for key, value in items.items():
            if value is None:
                fields.append(f"{parent}.{key}")
    if payload.population_density_per_km2 is None:
        fields.append("population_density_per_km2")
    return fields


def _power_grid_risk(payload: RiskAssessmentRequest) -> SystemComputation:
    power = payload.power_grid
    weather = payload.weather

    current_load = _num(power.current_load_percent)
    peak_load = _num(power.peak_load_percent)
    history = _num(power.historical_failure_rate_percent)
    transformer_age = _num(power.transformer_age_years)
    heatwave = bool(weather.heatwave_alert)

    history_norm = _cap(history * 4.0)
    transformer_age_norm = _age_norm(transformer_age, 40.0)
    heatwave_norm = 100.0 if heatwave else 0.0

    base = (
        0.32 * current_load
        + 0.22 * peak_load
        + 0.18 * transformer_age_norm
        + 0.18 * history_norm
        + 0.10 * heatwave_norm
    )

    drivers: list[str] = []
    high_factors = 0
    additive_bonus = 0.0

    if current_load > 85:
        drivers.append("current_load_percent>85")
        additive_bonus += 8.0
        high_factors += 1
    if peak_load > 90:
        drivers.append("peak_load_percent>90")
        additive_bonus += 7.0
        high_factors += 1
    if transformer_age > 20:
        drivers.append("transformer_age_years>20")
        additive_bonus += 6.0
        high_factors += 1
    if heatwave:
        drivers.append("heatwave_alert=true")
        additive_bonus += 5.0
        high_factors += 1
    if history > 15:
        drivers.append("historical_failure_rate_percent>15")
        additive_bonus += 6.0
        high_factors += 1

    compound_flags: list[str] = []
    score = base + additive_bonus
    compound_mult = _compound_multiplier(high_factors)
    if compound_mult > 1.0:
        compound_flags.append(f"multi_factor_amplification_{_compound_percent(compound_mult)}pct")
        score *= compound_mult
    if heatwave and current_load > 85:
        compound_flags.append("heatwave_plus_high_load_extra_amplification")
        score *= 1.08
    if not drivers:
        drivers.append("baseline_load_and_asset_profile")

    return SystemComputation(
        score=_cap(score),
        drivers=drivers,
        compound_flags=compound_flags,
        high_factor_count=high_factors,
    )


def _transformer_overload_risk(payload: RiskAssessmentRequest) -> SystemComputation:
    power = payload.power_grid
    weather = payload.weather

    current_load = _num(power.current_load_percent)
    peak_load = _num(power.peak_load_percent)
    history = _num(power.historical_failure_rate_percent)
    transformer_age = _num(power.transformer_age_years)
    heatwave = bool(weather.heatwave_alert)

    age_norm = _age_norm(transformer_age, 35.0)
    history_norm = _cap(history * 3.5)

    base = 0.38 * peak_load + 0.34 * current_load + 0.20 * age_norm + 0.08 * history_norm

    drivers: list[str] = []
    high_factors = 0
    additive_bonus = 0.0

    if peak_load > 90:
        drivers.append("peak_load_percent>90")
        additive_bonus += 10.0
        high_factors += 1
    if current_load > 85:
        drivers.append("current_load_percent>85")
        additive_bonus += 8.0
        high_factors += 1
    if transformer_age > 20:
        drivers.append("transformer_age_years>20")
        additive_bonus += 9.0
        high_factors += 1
    if history > 15:
        drivers.append("historical_failure_rate_percent>15")
        additive_bonus += 4.0
        high_factors += 1
    if heatwave:
        drivers.append("heatwave_alert=true")
        additive_bonus += 4.0
        high_factors += 1

    score = base + additive_bonus
    compound_flags: list[str] = []
    compound_mult = _compound_multiplier(high_factors)
    if compound_mult > 1.0:
        compound_flags.append(
            f"overload_intersection_amplification_{_compound_percent(compound_mult)}pct"
        )
        score *= compound_mult
    if peak_load > 90 and transformer_age > 20:
        compound_flags.append("aging_assets_under_peak_demand")

    if not drivers:
        drivers.append("baseline_transformer_load_profile")

    return SystemComputation(
        score=_cap(score),
        drivers=drivers,
        compound_flags=compound_flags,
        high_factor_count=high_factors,
    )


def _water_pipeline_risk(payload: RiskAssessmentRequest) -> SystemComputation:
    water = payload.water_system
    weather = payload.weather

    pressure_var = _num(water.pressure_variance_percent)
    age = _num(water.pipeline_age_years)
    repairs = _num(water.recent_repairs_last_30_days)
    rainfall = _num(weather.rainfall_mm)

    age_norm = _age_norm(age, 60.0)
    rainfall_norm = _cap((rainfall / 80.0) * 100.0)
    repairs_norm = _cap((repairs / 10.0) * 100.0)

    base = 0.36 * pressure_var + 0.28 * age_norm + 0.22 * rainfall_norm + 0.14 * repairs_norm

    drivers: list[str] = []
    high_factors = 0
    additive_bonus = 0.0

    if pressure_var > 30:
        drivers.append("pressure_variance_percent>30")
        additive_bonus += 10.0
        high_factors += 1
    if age > 25:
        drivers.append("pipeline_age_years>25")
        additive_bonus += 9.0
        high_factors += 1
    if rainfall > 40:
        drivers.append("rainfall_mm>40")
        additive_bonus += 8.0
        high_factors += 1
    if repairs > 3:
        drivers.append("recent_repairs_last_30_days>3")
        additive_bonus += 7.0
        high_factors += 1

    score = base + additive_bonus
    compound_flags: list[str] = []
    compound_mult = _compound_multiplier(high_factors)
    if compound_mult > 1.0:
        compound_flags.append(
            f"water_stress_compound_amplification_{_compound_percent(compound_mult)}pct"
        )
        score *= compound_mult
    if rainfall > 40 and pressure_var > 30:
        compound_flags.append("storm_pressure_instability_intersection")

    if not drivers:
        drivers.append("baseline_hydraulic_and_maintenance_profile")

    return SystemComputation(
        score=_cap(score),
        drivers=drivers,
        compound_flags=compound_flags,
        high_factor_count=high_factors,
    )


def _traffic_risk(payload: RiskAssessmentRequest) -> SystemComputation:
    traffic = payload.traffic_system
    congestion = _num(traffic.congestion_index)
    signal_failures = _num(traffic.signal_failure_reports_last_7_days)
    density = _num(payload.population_density_per_km2)

    signal_norm = _cap((signal_failures / 15.0) * 100.0)
    density_norm = _cap((density / 20000.0) * 100.0)

    base = 0.48 * congestion + 0.30 * signal_norm + 0.22 * density_norm

    drivers: list[str] = []
    high_factors = 0
    additive_bonus = 0.0

    if congestion > 75:
        drivers.append("congestion_index>75")
        additive_bonus += 10.0
        high_factors += 1
    if signal_failures > 5:
        drivers.append("signal_failure_reports_last_7_days>5")
        additive_bonus += 8.0
        high_factors += 1
    if density > 10000:
        drivers.append("population_density_per_km2>10000")
        additive_bonus += 7.0
        high_factors += 1

    score = base + additive_bonus
    compound_flags: list[str] = []
    compound_mult = _compound_multiplier(high_factors)
    if compound_mult > 1.0:
        compound_flags.append(f"traffic_compound_amplification_{_compound_percent(compound_mult)}pct")
        score *= compound_mult

    if not drivers:
        drivers.append("baseline_traffic_and_signal_profile")

    return SystemComputation(
        score=_cap(score),
        drivers=drivers,
        compound_flags=compound_flags,
        high_factor_count=high_factors,
    )


def _confidence_score(payload: RiskAssessmentRequest, missing_fields: list[str]) -> int:
    confidence = 95.0
    confidence -= len(missing_fields) * 5.0

    history = payload.power_grid.historical_failure_rate_percent
    if history is None or history < 5:
        confidence -= 10.0

    trend_signals = 0
    if payload.weather.heatwave_alert:
        trend_signals += 1
    if _num(payload.weather.rainfall_mm) > 20:
        trend_signals += 1
    if _num(payload.power_grid.current_load_percent) > 80:
        trend_signals += 1
    if _num(payload.traffic_system.congestion_index) > 70:
        trend_signals += 1
    if _num(payload.water_system.recent_repairs_last_30_days) > 2:
        trend_signals += 1
    if trend_signals == 0:
        confidence -= 8.0

    return int(round(_cap(confidence, 0, 100)))


def _outlook_24h(overall_level: str, systems: dict[str, SystemComputation]) -> str:
    ranked = sorted(systems.items(), key=lambda x: x[1].score, reverse=True)
    top_two = ", ".join(name.replace("_", " ") for name, _ in ranked[:2])

    if overall_level == "LOW":
        return f"Low citywide stress; maintain routine monitoring with focus on {top_two}."
    if overall_level == "MODERATE":
        return f"Elevated monitoring required in next 24 hours, centered on {top_two}."
    if overall_level == "HIGH":
        return f"Pre-failure posture detected; activate standby crews for {top_two} within 24 hours."
    return f"Critical posture; initiate emergency response and staged load reduction for {top_two} immediately."


def _projection_72h(payload: RiskAssessmentRequest, overall_score: float) -> tuple[float, str]:
    delta = 0.0

    heatwave = bool(payload.weather.heatwave_alert)
    current_load = _num(payload.power_grid.current_load_percent)
    peak_load = _num(payload.power_grid.peak_load_percent)
    rainfall = _num(payload.weather.rainfall_mm)
    pressure_var = _num(payload.water_system.pressure_variance_percent)
    congestion = _num(payload.traffic_system.congestion_index)
    signal_failures = _num(payload.traffic_system.signal_failure_reports_last_7_days)
    repairs = _num(payload.water_system.recent_repairs_last_30_days)
    density = _num(payload.population_density_per_km2)
    transformer_age = _num(payload.power_grid.transformer_age_years)
    pipeline_age = _num(payload.water_system.pipeline_age_years)

    if heatwave and current_load > 85:
        delta += 8.0
    elif heatwave:
        delta += 4.0

    if peak_load > 90:
        delta += 3.0
    if rainfall > 40:
        delta += 5.0
    if pressure_var > 30:
        delta += 2.0
    if congestion > 75:
        delta += 3.0
    if signal_failures > 5:
        delta += 2.0
    if repairs > 3:
        delta += 2.0
    if density > 10000 and (transformer_age > 20 or pipeline_age > 25):
        delta += 3.0

    projected = _cap(overall_score + delta)
    projected_level = classify_risk(projected)

    if projected > overall_score + 5:
        direction = "Escalating"
    elif projected < overall_score - 5:
        direction = "Improving"
    else:
        direction = "Stable-to-elevated"

    message = (
        f"{direction} trajectory with projected score {projected:.1f} ({projected_level}) by 72 hours "
        "based on current stress indicators."
    )
    return projected, message


def _priority_zones(
    systems: dict[str, SystemComputation],
    payload: RiskAssessmentRequest,
) -> list[str]:
    zones: list[str] = []
    if systems["power_grid"].score >= 66:
        zones.append("Substation clusters with sustained grid load above 85%")
    if systems["transformer_overload"].score >= 66:
        zones.append("Aging transformer corridors where age exceeds 20 years")
    if systems["water_pipeline"].score >= 66:
        zones.append("Legacy water mains with pressure variance above 30%")
    if systems["traffic_infrastructure"].score >= 66:
        zones.append("High-congestion junctions with repeated signal failures")

    density = _num(payload.population_density_per_km2)
    age_intersection = _num(payload.power_grid.transformer_age_years) > 20 or _num(
        payload.water_system.pipeline_age_years
    ) > 25
    if density > 10000 and age_intersection:
        zones.append("Dense districts with overlapping aging power and water assets")

    if not zones:
        zones.append("Citywide monitoring zones with no immediate critical hotspot")

    return zones


def run_risk_assessment(payload: RiskAssessmentRequest) -> RiskAssessmentResponse:
    missing_fields = _build_missing_fields(payload)

    power = _power_grid_risk(payload)
    transformer = _transformer_overload_risk(payload)
    water = _water_pipeline_risk(payload)
    traffic = _traffic_risk(payload)

    cascading_failure_risks: list[str] = []
    cascade_multiplier = 1.0
    traffic_adjustment = 0.0

    if power.score >= 66:
        impact = min(14.0, 5.0 + 0.12 * (power.score - 66.0))
        traffic_adjustment += impact
        cascade_multiplier += 0.03
        cascading_failure_risks.append(
            f"power_to_traffic_signal_dependency (impact +{impact:.1f} traffic risk)"
        )

    density = _num(payload.population_density_per_km2)
    if water.score >= 66 and density > 10000:
        impact = min(12.0, 4.0 + 0.10 * (water.score - 66.0))
        traffic_adjustment += impact
        cascade_multiplier += 0.03
        cascading_failure_risks.append(
            f"water_burst_to_traffic_disruption_in_dense_zones (impact +{impact:.1f} traffic risk)"
        )

    if _num(payload.traffic_system.congestion_index) > 75:
        cascade_multiplier += 0.02
        cascading_failure_risks.append("extreme_congestion_emergency_response_delay")

    age_intersection = _num(payload.power_grid.transformer_age_years) > 20 or _num(
        payload.water_system.pipeline_age_years
    ) > 25
    if density > 10000 and age_intersection:
        cascade_multiplier += 0.05
        cascading_failure_risks.append("population_density_plus_aging_infrastructure_amplification")

    traffic.score = _cap(traffic.score + traffic_adjustment)
    if traffic_adjustment > 0:
        traffic.compound_flags.append(f"cascading_adjustment_plus_{traffic_adjustment:.1f}")

    systems = {
        "power_grid": power,
        "transformer_overload": transformer,
        "water_pipeline": water,
        "traffic_infrastructure": traffic,
    }

    weighted = (
        0.33 * power.score
        + 0.17 * transformer.score
        + 0.25 * water.score
        + 0.25 * traffic.score
    )
    overall_score = _cap(weighted * cascade_multiplier)
    overall_level = classify_risk(overall_score)

    hour_24_outlook = _outlook_24h(overall_level, systems)
    _, hour_72_projection = _projection_72h(payload, overall_score)
    confidence_score = _confidence_score(payload, missing_fields)
    priority_zones = _priority_zones(systems, payload)

    systems_for_summary = {
        key: {
            "risk_score": value.score,
            "risk_level": classify_risk(value.score),
            "primary_drivers": value.drivers,
            "compound_flags": value.compound_flags,
        }
        for key, value in systems.items()
    }

    executive_summary = generate_executive_summary(
        city=payload.city,
        overall_score=overall_score,
        overall_level=overall_level,
        systems=systems_for_summary,
        cascading_failure_risks=cascading_failure_risks,
        confidence_score=confidence_score,
        missing_fields=missing_fields,
    )

    response_systems = {
        key: SystemRisk(
            risk_score=round(value.score, 2),
            risk_level=classify_risk(value.score),
            primary_drivers=value.drivers,
            compound_flags=value.compound_flags,
        )
        for key, value in systems.items()
    }

    return RiskAssessmentResponse(
        city=payload.city,
        timestamp=payload.timestamp,
        overall_risk=OverallRisk(
            score=round(overall_score, 2),
            level=overall_level,
            **{
                "24_hour_outlook": hour_24_outlook,
                "72_hour_projection": hour_72_projection,
            },
        ),
        systems=response_systems,
        cascading_failure_risks=cascading_failure_risks,
        priority_intervention_zones=priority_zones,
        executive_summary=executive_summary,
        confidence_score=confidence_score,
    )


def apply_ml_overlay(
    assessment: RiskAssessmentResponse,
    ml_probabilities: dict[str, float],
    ml_blend_weight: float = 0.55,
    ml_quality_score: float | None = None,
) -> RiskAssessmentResponse:
    if not ml_probabilities:
        return assessment

    blend = _cap(ml_blend_weight, 0.0, 1.0)
    rule_weight = 1.0 - blend
    target_to_system = {
        "power_failure_72h_label": "power_grid",
        "transformer_overload_72h_label": "transformer_overload",
        "water_failure_72h_label": "water_pipeline",
        "traffic_failure_72h_label": "traffic_infrastructure",
    }

    for target, system_key in target_to_system.items():
        if target not in ml_probabilities:
            continue
        system = assessment.systems[system_key]
        rule_score = float(system.risk_score)
        ml_score = _cap(float(ml_probabilities[target]) * 100.0)
        blended = _cap(rule_weight * rule_score + blend * ml_score)
        system.risk_score = round(blended, 2)
        system.risk_level = classify_risk(blended)
        system.compound_flags.append(
            f"ml_overlay_{target}={ml_score:.1f}_blend={blend:.2f}"
        )

    overall_prob = ml_probabilities.get("overall_failure_72h_label")
    if overall_prob is not None:
        rule_overall = float(assessment.overall_risk.score)
        ml_overall = _cap(float(overall_prob) * 100.0)
        blended_overall = _cap(rule_weight * rule_overall + blend * ml_overall)
        assessment.overall_risk.score = round(blended_overall, 2)
        assessment.overall_risk.level = classify_risk(blended_overall)

    if ml_quality_score is not None:
        quality = _cap(ml_quality_score * 100.0)
        confidence_shift = int(round((quality - 50.0) / 10.0))
        assessment.confidence_score = int(_cap(assessment.confidence_score + confidence_shift))
        assessment.executive_summary = (
            f"{assessment.executive_summary} ML overlay active with calibrated macro quality "
            f"score {quality:.1f}/100."
        )
    else:
        assessment.executive_summary = (
            f"{assessment.executive_summary} ML overlay active for probability calibration."
        )

    return assessment
