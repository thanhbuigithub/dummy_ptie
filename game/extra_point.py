from enum import Enum, auto

class ExtraPointType(Enum):
    NONE = auto()
    NORMAL_KNOCK = auto()
    DARK_KNOCK = auto()
    SUIT_KNOCK = auto()
    DUMMY_KNOCK = auto()
    BURN_STUPID = auto()
    STUPID_STUPID = auto()
    PICK_HEAD = auto()
    DUMMY_STUPID = auto()
    TAUNTING_STUPID = auto()
    SHOW_SPETO = auto()
    BE_LAYOFF_SPETO = auto()


class ExtraPointValueType(Enum):
    CONSTANT = 0
    MULTIPLE = 1


class ExtraPoint:
    type: ExtraPointType
    owner: int
    value: int
    value_type: ExtraPointValueType

    def __init__(self, t: ExtraPointType, owner: int):
        self.type = t
        self.owner = owner
        self.value = 0
        self.value_type = ExtraPointValueType.CONSTANT
        self._update_value()

    def __str__(self):
        return f"{self.type.name} {self.owner} {self.value} {self.value_type.name}"

    def _update_value(self):
        match self.type:
            case ExtraPointType.NONE:
                self.value = 0
                self.value_type = ExtraPointValueType.CONSTANT
            case ExtraPointType.NORMAL_KNOCK | ExtraPointType.DUMMY_KNOCK | ExtraPointType.PICK_HEAD | ExtraPointType.SHOW_SPETO:
                self.value = 50
                self.value_type = ExtraPointValueType.CONSTANT
            case ExtraPointType.DARK_KNOCK | ExtraPointType.SUIT_KNOCK | ExtraPointType.BURN_STUPID:
                self.value = 2
                self.value_type = ExtraPointValueType.MULTIPLE
            case _:
                self.value = -50
                self.value_type = ExtraPointValueType.CONSTANT

    def update_score(self, score: int):
        if self.type == ExtraPointType.BURN_STUPID and score > 0:
            score = 0
        if self.value_type == ExtraPointValueType.CONSTANT:
            score += self.value
        else:
            score = self.value * score
        return score

