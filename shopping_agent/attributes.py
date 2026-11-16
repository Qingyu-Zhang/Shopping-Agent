from __future__ import annotations

import json
import math
import re
from operator import eq, ge, gt, le, lt
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, StrictFloat, StrictInt, StrictStr, ValidationError, WithJsonSchema, model_validator

from .core import Input

SCHEMA = json.loads(Path(__file__).with_name("attributes.json").read_text(encoding="utf-8"))
CATEGORIES = {name: data["query"] for name, data in SCHEMA["categories"].items() if data["query"]}
ATTRIBUTES = SCHEMA["attributes"]
COMPARISONS = {"eq": eq, "gte": ge, "lte": le, "gt": gt, "lt": lt}


def valid_category(value):
    if value not in SCHEMA["categories"]:
        raise ValueError("Use a category from the catalog.")
    return value


Category = Annotated[str, AfterValidator(valid_category),
                     WithJsonSchema({"type": "string", "enum": list(SCHEMA["categories"])})]


def category_attributes(category):
    names = SCHEMA["categories"].get(category, {}).get("attributes", [])
    return {name: spec for name, spec in ATTRIBUTES.items() if name in names or spec.get("all_categories")}


def attribute_schema():
    return {category: {
        name: {key: val for key, val in spec.items() if key in {"type", "unit", "min", "max"}}
        for name, spec in category_attributes(category).items()
    } for category in SCHEMA["categories"]}


def normalize_value(spec, value, *, parse_number=False):
    if spec["type"] == "number":
        if parse_number and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None
        if type(value) not in (int, float) or type(value) is float and not math.isfinite(value):
            return None
        if not spec.get("min", -math.inf) <= value <= spec.get("max", math.inf):
            return None
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    key = " ".join(value.casefold().split())
    return spec.get("aliases", {}).get(key, key)


class AttributeTarget(Input):
    category: Category
    attribute: str

    @model_validator(mode="after")
    def known_attribute(self):
        if self.attribute not in category_attributes(self.category):
            raise ValueError("Use an attribute defined for the selected category.")
        return self


Scalar = StrictStr | StrictInt | StrictFloat


class Condition(AttributeTarget):
    operator: Literal["eq", "gte", "lte", "gt", "lt", "in", "contains_any", "contains_all"]
    value: Scalar | list[Scalar]
    required: bool = True

    @model_validator(mode="after")
    def typed_condition(self):
        spec = ATTRIBUTES[self.attribute]
        if spec["type"] == "set":
            allowed = {"contains_any", "contains_all"}
        elif spec["type"] == "number":
            allowed = set(COMPARISONS) | {"in"}
        else:
            allowed = {"eq", "in"}
        plural = self.operator in {"in", "contains_any", "contains_all"}
        if self.operator not in allowed or plural != isinstance(self.value, list):
            raise ValueError("The operator and value must match the catalog attribute type.")
        values = self.value if plural else [self.value]
        normalized = [normalize_value(spec, value) for value in values]
        if not 1 <= len(normalized) <= 15 or any(value is None for value in normalized):
            raise ValueError("The value is outside the catalog attribute definition.")
        self.value = normalized if plural else normalized[0]
        return self

    def matches(self, actual):
        spec = ATTRIBUTES[self.attribute]
        if spec["type"] == "set":
            if not isinstance(actual, list) or not actual:
                return None
            values = {normalize_value(spec, value) for value in actual}
            values.discard(None)
            if not values:
                return None
            return (bool(values.intersection(self.value)) if self.operator == "contains_any"
                    else values.issuperset(self.value))
        actual = normalize_value(spec, actual)
        if actual is None:
            return None
        return actual in self.value if self.operator == "in" else COMPARISONS[self.operator](actual, self.value)


def _category_phrases(text, categories):
    phrases = {}
    if len(categories) < 2:
        return phrases
    for phrase in re.split(r"\s+(?:and|plus)\s+(?!\d)|;\s*", text):
        mentioned = [category for category in categories if category in CATEGORIES
                     and re.search(CATEGORIES[category], phrase)]
        if len(mentioned) == 1:
            phrases.setdefault(mentioned[0], []).append(phrase)
    return {category: " ".join(parts) for category, parts in phrases.items()} if len(phrases) > 1 else {}


def detect_conditions(text, categories):
    scoped_phrases = _category_phrases(text, categories)
    for category in categories:
        request = scoped_phrases.get(category, text)
        for name, spec in category_attributes(category).items():
            for rule in spec.get("rules", ()):
                matches = list(re.finditer(rule["pattern"], request))
                if not matches:
                    continue
                match = matches[0]
                prefix = request[max(0, match.start()-100):match.start()]
                clause = re.split(r"[,;!?]|(?<!\d)\.(?!\d)|\bbut\b", prefix)[-1]
                if re.search(r"\b(?:no|not|avoid|without)\b\s*$", clause):
                    break
                required = not re.search(r"\b(prefer|preferably|ideally|nice to have|if possible)\b", clause)
                value = normalize_value(spec, match["value"], parse_number=True)
                operator = rule["operator"]
                lower_bound = None
                if spec["type"] == "number":
                    before_value = request[max(0, match.start("value")-65):match.start("value")]
                    if found := re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s*$", before_value):
                        lower_bound = normalize_value(spec, found[1], parse_number=True)
                        operator = "lte"
                    elif re.search(r"\b(?:at most|up to|no more than)\s*$", before_value):
                        operator = "lte"
                    elif re.search(r"\b(?:under|below|less than)\s*$", before_value):
                        operator = "lt"
                    elif re.search(r"\b(?:more than|greater than|above|over)\s*$", before_value):
                        operator = "gt"
                    elif re.search(r"\b(?:at least|no less than)\s*$", before_value):
                        operator = "gte"
                if spec["type"] == "set":
                    value = [value]
                elif operator == "eq" and len(matches) > 1 and request[match.end():matches[1].start()].strip() == "or":
                    operator, value = "in", [value, normalize_value(spec, matches[1]["value"], parse_number=True)]
                try:
                    if lower_bound is not None:
                        yield Condition(category=category, attribute=name, operator="gte",
                                        value=lower_bound, required=required)
                    yield Condition(category=category, attribute=name, operator=operator, value=value, required=required)
                except ValidationError:
                    pass
                break


def metadata_attributes(category, details):
    hints = {}
    for name, spec in category_attributes(category).items():
        for key in spec.get("metadata", ()):
            value = details.get(key)
            if spec["type"] == "set" and isinstance(value, str):
                values = [normalize_value(spec, part) for part in re.split(r"[,/;]", value)]
                normalized = [part for part in values if part]
            else:
                normalized = normalize_value(spec, value, parse_number=True)
            if normalized is not None and normalized != []:
                hints[name] = {"value": normalized, "source_field": key}
                break
    return hints


def active_conditions(goal, category):
    conditions = [Condition.model_validate(raw) for raw in goal.get("conditions", ()) if raw["category"] == category]
    names = {condition.attribute for condition in conditions}
    ceiling = goal.get("max_weights", {}).get(category)
    if ceiling is not None and "weight_g" not in names:
        conditions.append(Condition(category=category, attribute="weight_g", operator="lte", value=ceiling))
    memory = goal.get("min_ram")
    if category == "laptop" and memory and "ram_gb" not in names:
        conditions.append(Condition(category=category, attribute="ram_gb", operator="gte", value=memory))
    return conditions
