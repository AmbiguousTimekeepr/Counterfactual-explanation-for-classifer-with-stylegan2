"""Attribute configuration for the ResNet18+CBAM classifier variant."""

ATTR_MAP = {
    "Bald": "Bald",
    "Bangs": "Bangs",
    "Black_Hair": "Black Hair",
    "Blond_Hair": "Blond Hair",
    "Brown_Hair": "Brown Hair",
    "Bushy_Eyebrows": "Bushy Eyebrows",
    "Eyeglasses": "Eyeglasses",
    "Male": "Gender (Male)",
    "Mouth_Slightly_Open": "Open Mouth",
    "Mustache": "Mustache",
    "Pale_Skin": "Pale Skin",
    "Young": "Age (Young)",
}

SELECTED_ATTRIBUTES = list(ATTR_MAP.keys())
ATTRIBUTE_DISPLAY_NAMES = list(ATTR_MAP.values())

__all__ = [
    "ATTR_MAP",
    "SELECTED_ATTRIBUTES",
    "ATTRIBUTE_DISPLAY_NAMES",
]
