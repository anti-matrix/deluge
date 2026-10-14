"""Phase 2, capacity: carrying capacity per cell for each subsistence mode."""

from __future__ import annotations


def run(model) -> None:
    model.capacity = model.carrying_capacity.compute(
        model.temperature, model.rainfall, model.is_sea, model.is_coast)
    model.k_forage = float(model.capacity["forage"].sum())
    model.k_herd = float(model.capacity["herd"].sum())
    model.k_farm = float(model.capacity["farm"].sum())
