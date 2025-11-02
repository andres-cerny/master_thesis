from enum import Enum
import pandas as pd

class CESTStart(Enum):
    _2020 = pd.Timestamp("2020-03-29")
    _2021 = pd.Timestamp("2021-03-28")
    _2022 = pd.Timestamp("2022-03-27")
    _2023 = pd.Timestamp("2023-03-26")
    _2024 = pd.Timestamp("2024-03-31")
    _2025 = pd.Timestamp("2025-03-30")
    _2026 = pd.Timestamp("2026-03-29")
    _2027 = pd.Timestamp("2027-03-28")
    _2028 = pd.Timestamp("2028-03-26")
    _2029 = pd.Timestamp("2029-03-25")
    _2030 = pd.Timestamp("2030-03-31")

class CESTEnd(Enum):
    _2020 = pd.Timestamp("2020-10-25")
    _2021 = pd.Timestamp("2021-10-31")
    _2022 = pd.Timestamp("2022-10-30")
    _2023 = pd.Timestamp("2023-10-29")
    _2024 = pd.Timestamp("2024-10-27")
    _2025 = pd.Timestamp("2025-10-26")
    _2026 = pd.Timestamp("2026-10-25")
    _2027 = pd.Timestamp("2027-10-31")
    _2028 = pd.Timestamp("2028-10-29")
    _2029 = pd.Timestamp("2029-10-28")
    _2030 = pd.Timestamp("2030-10-27")

def is_cest_start(date_obj):
    """
    Check if a date is a CEST start date.
    
    Args:
        date_obj: pandas Timestamp, datetime object, or string convertible to Timestamp
    
    Returns:
        bool: True if date is a CEST start date, False otherwise
    """
    date_obj = pd.Timestamp(date_obj).date()
    return any(date_obj == item.value.date() for item in CESTStart)

def is_cest_end(date_obj):
    """
    Check if a date is a CEST end date.
    
    Args:
        date_obj: pandas Timestamp, datetime object, or string convertible to Timestamp
    
    Returns:
        bool: True if date is a CEST end date, False otherwise
    """
    date_obj = pd.Timestamp(date_obj).date()
    return any(date_obj == item.value.date() for item in CESTEnd)

