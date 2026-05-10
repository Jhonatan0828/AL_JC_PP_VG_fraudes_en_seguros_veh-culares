"""
Tests unitarios para el paquete fraud_detection.
Ejecutar:  pytest tests/
"""
import pandas as pd
import pytest

from src.fraud_detection.mappings import (
    EDA_FILTERS, ORDINAL_MAPPINGS, TOP_PREDICTORS,
)
from src.fraud_detection.prediction import (
    build_input_dataframe, detect_risk_factors,
)


# -------- Tests sobre mappings --------

def test_ordinal_mappings_have_consistent_values():
    """Todos los valores de los mapeos deben ser enteros."""
    for col, mapping in ORDINAL_MAPPINGS.items():
        for key, value in mapping.items():
            assert isinstance(value, int), (
                f"{col}['{key}'] no es entero: {value}"
            )


def test_eda_filters_is_non_empty_list():
    assert isinstance(EDA_FILTERS, list)
    assert len(EDA_FILTERS) > 0
    assert all(isinstance(f, str) for f in EDA_FILTERS)


def test_top_predictors_have_required_keys():
    required_keys = {"Variable", "Tipo", "Importancia", "Interpretación"}
    for predictor in TOP_PREDICTORS:
        assert required_keys.issubset(predictor.keys())


# -------- Tests sobre prediction --------

def test_build_input_dataframe_applies_ordinal_mappings():
    """El mapeo ordinal debe convertir strings a enteros."""
    form_data = {
        "VehiclePrice": "less than 20000",
        "PastNumberOfClaims": "more than 4",
        "Fault": "Policy Holder",  # Categórica, no se mapea
    }
    df = build_input_dataframe(form_data)

    assert df["VehiclePrice"].iloc[0] == 0
    assert df["PastNumberOfClaims"].iloc[0] == 3
    assert df["Fault"].iloc[0] == "Policy Holder"


def test_build_input_dataframe_adds_default_fields():
    """Campos fijos como WeekOfMonth y RepNumber deben tener valor por defecto."""
    df = build_input_dataframe({"Age": 30})
    assert df["WeekOfMonth"].iloc[0] == 3
    assert df["WeekOfMonthClaimed"].iloc[0] == 3
    assert df["RepNumber"].iloc[0] == 12


def test_detect_risk_factors_high_risk_case():
    """Caso de alto riesgo: debe detectar múltiples factores."""
    form_data = {
        "Fault": "Policy Holder",
        "BasePolicy": "All Perils",
        "PastNumberOfClaims": "more than 4",
        "AddressChange_Claim": "under 6 months",
    }
    factors = detect_risk_factors(form_data)
    assert len(factors) == 4


def test_detect_risk_factors_low_risk_case():
    """Caso de bajo riesgo: no debe detectar factores."""
    form_data = {
        "Fault": "Third Party",
        "BasePolicy": "Liability",
        "PastNumberOfClaims": "none",
        "AddressChange_Claim": "no change",
    }
    factors = detect_risk_factors(form_data)
    assert len(factors) == 0


def test_detect_risk_factors_partial():
    """Caso parcial: solo algunos factores activos."""
    form_data = {
        "Fault": "Policy Holder",
        "BasePolicy": "Liability",
        "PastNumberOfClaims": "none",
        "AddressChange_Claim": "no change",
    }
    factors = detect_risk_factors(form_data)
    assert len(factors) == 1
