from typing import Iterator

from ocelot.cpbd.elements.optic_element import OpticElement
from ocelot.cpbd.magnetic_lattice import flatten

class Line:
    def __init__(self, elements: list[OpticElement] | None = None):
        self._elements = list(flatten(elements)) or []
        self._names = {}
        for element in self._elements:
            self._names[element.id] = element

    def __getitem__(self, key: int | str) -> OpticElement:
        if isinstance(key, int):
            return self.elements[key]
        elif isinstance(key, str):
            return self._names[key]
        else:
            raise KeyError("Key must be int or str")
        
    def __iter__(self) -> Iterator[OpticElement]:
        return iter(self._elements)

    def __len__(self) -> int:
        return len(self._elements)

    def __repr__(self) -> str:
        return f"Line({self._elements})"

    # def append(self, element: OpticElement) -> None:
    #     try:
    #         ele = self[element.id]
    #     except KeyError:
    #         pass
    #     else:
    #         if ele is not element:
    #             raise ValueError("Unique element with this name already exists")

    #     self._elements.append(element)
    #     self._names[element.id] = element
            
