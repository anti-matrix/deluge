"""Phase 2, capacity: carrying capacity per cell for each subsistence mode.

Also ``farm_calendar``: farming capacity for settlements whose techniques soften drought
(sky observation and calendar), where a drought takes only ``drought_loss`` of the rain
it would otherwise take. Settlement-specific technique multipliers are applied in the
subsistence phase.
"""

from __future__ import annotations


def run(model) -> None:
    model.capacity = model.carrying_capacity.compute(
        model.temperature, model.rainfall, model.is_sea, model.is_coast)
    softening = float(model.knowledge.effect["drought_loss"].min())
    factor = model.shocks.rain_factor
    if softening < 1.0 and factor.min() < 1.0:
        rain = model.base_rain_flat * (1.0 - (1.0 - factor) * softening)
        model.capacity["farm_calendar"] = model.carrying_capacity.compute(
            model.temperature, rain, model.is_sea, model.is_coast)["farm"]
    else:
        model.capacity["farm_calendar"] = model.capacity["farm"]
    model.k_forage = float(model.capacity["forage"].sum())
    model.k_herd = float(model.capacity["herd"].sum())
    model.k_farm = float(model.capacity["farm"].sum())
