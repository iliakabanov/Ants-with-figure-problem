"""
Типы углов контура фигуры в системе тела.

Роза 0..7 (как в будущих лучах): 0=С, 1=СВ, 2=В, 3=ЮВ, 4=Ю, 5=ЮЗ, 6=З, 7=СЗ; шаг 45° по часовой от севера.
«Внешний» — выпуклая вершина силуэта; «внутренний» — вогнутая (вырез буквы Т).
Направление сектора — ближайшая к биссектрисе «пустоты»: снаружи у выпуклого, в вырез у вогнутого.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Индекс 0 = север (+y), далее по часовой (та же конвенция, что у compass8_dir_to_body_rad).
WIND8_NAMES_RU: tuple[str, ...] = ("С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ")


def _wind8_angle_rad(k: int) -> float:
    i = int(k) % 8
    return 0.5 * math.pi - i * (0.25 * math.pi)


def five_rose_indices_around(center_wind8: int) -> tuple[int, int, int, int, int]:
    """
    Пять направлений розы (0..7), ближайших к сектору угла ``center_wind8``:
    на два шага по часовой в обе стороны, например для СВ (1): СЗ, С, СВ, В, ЮВ → 7,0,1,2,3.
    """
    k = int(center_wind8) % 8
    return tuple((k + d) % 8 for d in (-2, -1, 0, 1, 2))


def _nearest_wind8(angle_rad: float) -> int:
    best_k = 0
    best_d = 1e9
    for k in range(8):
        ak = _wind8_angle_rad(k)
        d = (angle_rad - ak) % (2 * math.pi)
        d = min(d, 2 * math.pi - d)
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


def _polygon_vertices_ccw(vertices: list[tuple[float, float]]) -> bool:
    n = len(vertices)
    if n < 3:
        return True
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s >= 0.0


def _vertex_is_reflex(
    vertex_index: int,
    vertices: list[tuple[float, float]],
    *,
    poly_ccw: bool,
) -> bool:
    n = len(vertices)
    a = vertices[vertex_index - 1]
    b = vertices[vertex_index]
    c = vertices[(vertex_index + 1) % n]
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    cross = ux * vy - uy * vx
    if poly_ccw:
        return cross < 0.0
    return cross > 0.0


def _bisector_from_neighbor_unit_sum(vertex_index: int, vertices: list[tuple[float, float]]) -> float:
    """Биссектриса по сумме единичных B→A и B→C (направление «в материал» у выпуклого угла)."""
    n = len(vertices)
    a = vertices[vertex_index - 1]
    b = vertices[vertex_index]
    c = vertices[(vertex_index + 1) % n]
    dx1, dy1 = a[0] - b[0], a[1] - b[1]
    dx2, dy2 = c[0] - b[0], c[1] - b[1]
    l1 = math.hypot(dx1, dy1)
    l2 = math.hypot(dx2, dy2)
    if l1 < 1e-15 or l2 < 1e-15:
        return 0.0
    mx = dx1 / l1 + dx2 / l2
    my = dy1 / l1 + dy2 / l2
    if mx * mx + my * my < 1e-18:
        return 0.0
    return math.atan2(my, mx)


@dataclass(frozen=True, slots=True)
class FigureCornerLabel:
    """Один угол контура в системе тела фигуры."""

    external: bool
    """True — внешний (выпуклый) угол силуэта; False — внутренний (вогнутый, вырез)."""
    wind8: int
    """0..7: С, СВ, В, … СЗ (шаг 45° по часовой от севера)."""

    def wind_name_ru(self) -> str:
        return WIND8_NAMES_RU[self.wind8 % 8]

    def __str__(self) -> str:
        kind = "внешний" if self.external else "внутренний"
        return f"{kind} {self.wind_name_ru()}"


def compute_corner_labels_for_outline(
    outline_local: list[tuple[float, float]],
) -> tuple[FigureCornerLabel, ...]:
    """
    Для каждой вершины ``outline_local`` (тот же порядок, что обход контура) —
    тип угла и сектор розы.

    У выпуклого: сектор по биссектрисе **наружу** (в пустоту).
    У вогнутого: сектор по биссектрисе **в вырез** (противоположно «в материал»).
    """
    n = len(outline_local)
    if n == 0:
        return ()
    poly_ccw = _polygon_vertices_ccw(outline_local)
    out: list[FigureCornerLabel] = []
    for vi in range(n):
        interior = _bisector_from_neighbor_unit_sum(vi, outline_local)
        # Выпуклый: «воздух» снаружи силуэта; вогнутый: «воздух» в вырезе — в обоих случаях +π к биссектрисе в материал.
        air = (interior + math.pi) % (2 * math.pi)
        ext = not _vertex_is_reflex(vi, outline_local, poly_ccw=poly_ccw)
        out.append(FigureCornerLabel(external=ext, wind8=_nearest_wind8(air)))
    return tuple(out)
