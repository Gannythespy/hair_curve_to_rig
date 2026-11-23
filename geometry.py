"""Curve sampling and longitudinal skinning, independent of Blender operators."""

from bisect import bisect_left
from math import floor

from mathutils import Vector


class RigError(ValueError):
    """An input that cannot produce a reliable hair rig."""


def cumulative_lengths(points):
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + (b - a).length)
    return lengths


def sample_bezier(spline, matrix):
    """Adaptive world-space polyline, with exact endpoints and control-point joins."""
    controls = spline.bezier_points
    points = [matrix @ controls[0].co]
    scale = sum(
        (matrix @ b.co - matrix @ a.co).length
        + (matrix @ a.handle_right - matrix @ a.co).length
        + (matrix @ b.handle_left - matrix @ b.co).length
        for a, b in zip(controls, list(controls)[1:])
    )
    tolerance = max(scale * 1e-7, 1e-10)

    def subdivide(a, b, c, d, depth=0):
        polygon = (b - a).length + (c - b).length + (d - c).length
        if depth >= 14 or (depth >= 4 and polygon - (d - a).length <= tolerance):
            points.append(d)
            return
        ab, bc, cd = (a + b) / 2, (b + c) / 2, (c + d) / 2
        abc, bcd = (ab + bc) / 2, (bc + cd) / 2
        middle = (abc + bcd) / 2
        subdivide(a, ab, abc, middle, depth + 1)
        subdivide(middle, bcd, cd, d, depth + 1)

    for a, b in zip(controls, list(controls)[1:]):
        subdivide(matrix @ a.co, matrix @ a.handle_right,
                  matrix @ b.handle_left, matrix @ b.co)
    return points


def evenly_spaced(points, count):
    lengths = cumulative_lengths(points)
    total = lengths[-1]
    if total <= 1e-8:
        raise RigError("The hair path has zero length.")
    result = [points[0].copy()]
    for i in range(1, count):
        distance = total * i / count
        j = max(1, bisect_left(lengths, distance))
        span = lengths[j] - lengths[j - 1]
        result.append(points[j - 1].lerp(points[j], (distance - lengths[j - 1]) / span))
    result.append(points[-1].copy())
    if any((b - a).length <= max(total * 1e-8, 1e-8)
           for a, b in zip(result, result[1:])):
        raise RigError("A bone would have zero length. Increase the count or edit the path.")
    return result


def weights_at(fraction, count, skip_root_bones=0):
    """Return original segment indices; -1 denotes the fixed attachment bone.

    The entire skipped interval is rigidly attached. Blend into the first live
    bone only AFTER that interval, reaching full influence at its midpoint.
    With no skipped segments this is exactly the original weighting behavior.
    """
    if skip_root_bones:
        segment_position = min(max(fraction, 0.0), 1.0) * count
        if segment_position <= skip_root_bones:
            return ((-1, 1.0),)
        if segment_position < skip_root_bones + 0.5:
            blend = 2.0 * (segment_position - skip_root_bones)
            return ((-1, 1.0 - blend), (skip_root_bones, blend))
    position = min(max(fraction * count - 0.5, 0.0), count - 1.0)
    lower = int(floor(position))
    blend = position - lower
    if blend <= 1e-7 or lower == count - 1:
        return ((lower, 1.0),)
    if blend >= 1.0 - 1e-7:
        return ((lower + 1, 1.0),)
    return ((lower, 1.0 - blend), (lower + 1, blend))


def transported_normals(joints):
    """A rotation-minimizing frame avoids arbitrary roll flips at axis crossings."""
    directions = [(b - a).normalized() for a, b in zip(joints, joints[1:])]
    axes = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
    reference = min(axes, key=lambda axis: abs(axis.dot(directions[0])))
    normal = (reference - directions[0] * reference.dot(directions[0])).normalized()
    normals = [normal]
    for previous, direction in zip(directions, directions[1:]):
        normal = previous.rotation_difference(direction) @ normal
        normal = (normal - direction * normal.dot(direction)).normalized()
        normals.append(normal)
    return normals
