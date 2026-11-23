"""Native bevel conversion, transactional rig creation, and skinning."""

import re
from dataclasses import dataclass

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from .geometry import (
    RigError, cumulative_lengths, evenly_spaced, sample_bezier,
    transported_normals, weights_at,
)

TAG = "hair_curve_rig_version"


@dataclass
class PreparedLock:
    source: object
    spline_index: int
    count: int
    skip_root_bones: int
    reverse: bool
    joints: list
    label: str


def validate_source(source, spline_index, point_count=None):
    if source is None or source.type != 'CURVE':
        raise RigError("A source curve is missing. Load your selection again.")
    label = f"{source.name}, spline {spline_index + 1}"
    if spline_index >= len(source.data.splines):
        raise RigError(f"{label}: spline was removed; load your selection again.")
    curve = source.data
    spline = curve.splines[spline_index]
    if spline.type != 'BEZIER' or spline.use_cyclic_u or len(spline.bezier_points) < 2:
        raise RigError(f"{label}: use an open Bézier spline with at least two control points.")
    if point_count is not None and len(spline.bezier_points) != point_count:
        raise RigError(f"{label}: control-point count changed; load your selection again.")
    if curve.dimensions != '3D':
        raise RigError(f"{label}: set Curve Data > Shape > Dimensions to 3D.")
    if abs(source.matrix_world.determinant()) < 1e-12:
        raise RigError(f"{label}: object scale must be nonzero.")
    if any(mod.show_viewport or mod.show_render for mod in source.modifiers):
        raise RigError(f"{label}: active modifiers are not supported; use an unmodified curve copy.")
    if curve.shape_keys:
        raise RigError(f"{label}: shape keys are not supported in this version.")
    if curve.extrude != 0 or curve.offset != 0:
        raise RigError(f"{label}: set curve Extrude and Offset to zero.")
    if curve.bevel_factor_start != 0 or curve.bevel_factor_end != 1:
        raise RigError(f"{label}: use the full bevel range (Start 0, End 1).")
    profile = curve.bevel_object
    if curve.bevel_mode != 'OBJECT' or profile is None or profile.type != 'CURVE':
        raise RigError(f"{label}: choose a closed curve as the bevel object.")
    if profile == source or not profile.data.splines or any(
        not sp.use_cyclic_u for sp in profile.data.splines
    ):
        raise RigError(f"{label}: the bevel profile must consist of closed loops.")
    if any(mod.show_viewport or mod.show_render for mod in profile.modifiers):
        raise RigError(f"{label}: bevel-profile modifiers are not supported.")
    if curve.taper_object and any(
        mod.show_viewport or mod.show_render for mod in curve.taper_object.modifiers
    ):
        raise RigError(f"{label}: taper-object modifiers are not supported.")
    return spline


def prepare(context, settings):
    if context.mode != 'OBJECT':
        raise RigError("Switch to Object Mode before generating.")
    context.view_layer.update()
    rows = [row for row in settings.locks if row.enabled]
    if not rows:
        raise RigError("Load and enable at least one hair spline.")
    armature = None
    if settings.destination == 'EXISTING':
        armature = settings.armature
        if armature is None or armature.type != 'ARMATURE':
            raise RigError("Choose an existing armature.")
        if not armature.is_editable or not armature.data.is_editable:
            raise RigError("The destination armature must be locally editable.")
        if armature.data.users != 1:
            raise RigError("Make the destination armature data single-user first.")
        if armature.name not in context.view_layer.objects or not armature.visible_get():
            raise RigError("Show the destination armature in the current view layer first.")
        if armature.hide_select:
            raise RigError("Enable selection for the destination armature first.")
        if abs(armature.matrix_world.determinant()) < 1e-12:
            raise RigError("The destination armature scale must be nonzero.")
        parent = armature.data.bones.get(settings.parent_bone)
        if parent is None:
            raise RigError("Choose the parent bone, usually the character's head bone.")
        if armature.data.pose_position != 'REST':
            delta = armature.pose.bones[parent.name].matrix - parent.matrix_local
            if max(abs(value) for row in delta for value in row) > 1e-5:
                raise RigError("The parent bone is posed. Use the character's rest pose before generating.")
    prepared = []
    seen = set()
    for row in rows:
        spline = validate_source(row.source, row.spline_index, row.point_count)
        if not 0 <= row.skip_root_bones <= row.bone_count:
            raise RigError("Skip Root Bones must be between zero and the strand's bone count.")
        key = (row.source.as_pointer(), row.spline_index)
        if key in seen:
            raise RigError("The same spline is listed twice. Load your selection again.")
        seen.add(key)
        evaluated = row.source.evaluated_get(context.evaluated_depsgraph_get())
        points = sample_bezier(evaluated.data.splines[row.spline_index], evaluated.matrix_world)
        if row.reverse:
            points.reverse()
        joints = evenly_spaced(points, row.bone_count)
        prepared.append(PreparedLock(
            row.source, row.spline_index, row.bone_count, row.skip_root_bones, row.reverse, joints,
            f"{row.source.name} / spline {row.spline_index + 1}",
        ))
    return prepared, armature


def _evaluated_mesh(obj, context):
    context.view_layer.update()
    graph = context.evaluated_depsgraph_get()
    return bpy.data.meshes.new_from_object(
        obj.evaluated_get(graph), preserve_all_data_layers=True, depsgraph=graph,
    )


def _ordered_wire(mesh, start):
    neighbors = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        neighbors[a].append(b)
        neighbors[b].append(a)
    ends = [i for i, links in enumerate(neighbors) if len(links) == 1]
    if len(ends) != 2 or any(len(links) not in (1, 2) for links in neighbors):
        raise RigError("Blender did not produce one continuous open centerline.")
    current = min(ends, key=lambda i: (mesh.vertices[i].co - start).length_squared)
    previous = None
    ordered = []
    while current is not None:
        ordered.append(mesh.vertices[current].co.copy())
        following = [i for i in neighbors[current] if i != previous]
        previous, current = current, following[0] if following else None
        if len(ordered) > len(mesh.vertices):
            raise RigError("Unexpected cyclic centerline.")
    if len(ordered) != len(mesh.vertices):
        raise RigError("The evaluated centerline has disconnected geometry.")
    return ordered


def longitudinal_coordinates(mesh, wire_points, matrix):
    """Use Blender's native side UVs to associate cross-sections with path samples.

    Unlike nearest-bone weighting this stays ordered through tight curls and
    crossings. Caps have separate UVs/vertices, matched only to endpoint rings.
    Reject unexpected topology rather than assigning ambiguous weights.
    """
    if not mesh.uv_layers or not mesh.polygons:
        raise RigError("The bevel produced no usable surface/UVs. Check its profile and radius.")
    uv = mesh.uv_layers.active.data
    by_vertex = {}
    rings = set()
    for face in mesh.polygons:
        values = [uv[i].uv for i in face.loop_indices]
        # Native bevel side faces span both U (length) and V (profile).
        # Cap triangles have a constant V and must not supply length coordinates.
        if (max(v.x for v in values) - min(v.x for v in values) < 1e-8
                or max(v.y for v in values) - min(v.y for v in values) < 1e-8):
            continue
        for loop_index in face.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            key = round(uv[loop_index].uv.x, 7)
            if vertex_index in by_vertex and by_vertex[vertex_index] != key:
                raise RigError("A bevel vertex has inconsistent longitudinal UVs.")
            by_vertex[vertex_index] = key
            rings.add(key)
    rings = sorted(rings)
    if len(rings) != len(wire_points) or len(rings) < 2:
        raise RigError("Bevel cross-sections do not match the centerline; this curve layout is unsupported.")
    distances = cumulative_lengths([matrix @ p for p in wire_points])
    if distances[-1] <= 1e-8:
        raise RigError("The evaluated centerline has zero length.")
    fractions = {key: distance / distances[-1] for key, distance in zip(rings, distances)}
    result = [None] * len(mesh.vertices)
    endpoints = []
    for index, key in by_vertex.items():
        result[index] = fractions[key]
        if key in (rings[0], rings[-1]):
            endpoints.append(index)
    tree = KDTree(len(endpoints))
    for index in endpoints:
        tree.insert(mesh.vertices[index].co, index)
    tree.balance()
    tolerance = max(max((p - wire_points[0]).length for p in wire_points) * 1e-5, 1e-6)
    for vertex in mesh.vertices:
        if result[vertex.index] is not None:
            continue
        _, closest, distance = tree.find(vertex.co)
        if distance > tolerance:
            raise RigError("A cap vertex could not be associated with an endpoint ring.")
        result[vertex.index] = result[closest]
    return result


def convert_lock(context, lock, collection):
    """Use a private curve copy; never change the source or the shared profile."""
    data = lock.source.data.copy()
    temporary = bpy.data.objects.new("HairRig_TemporaryCurve", data)
    collection.objects.link(temporary)
    temporary.matrix_world = lock.source.matrix_world.copy()
    data.animation_data_clear()
    for index, slot in enumerate(lock.source.material_slots):
        if slot.link == 'OBJECT':
            data.materials[index] = slot.material
    surface = wire = None
    try:
        # Remove other splines in reverse order, retaining all settings on this one.
        for index in reversed(range(len(data.splines))):
            if index != lock.spline_index:
                data.splines.remove(data.splines[index])
        surface = _evaluated_mesh(temporary, context)
        data.bevel_object = None
        data.bevel_depth = 0
        data.bevel_mode = 'ROUND'
        data.taper_object = None
        data.use_fill_caps = False
        wire = _evaluated_mesh(temporary, context)
        points = _ordered_wire(wire, data.splines[0].bezier_points[0].co)
        fractions = longitudinal_coordinates(surface, points, temporary.matrix_world)
        if lock.reverse:
            fractions = [1.0 - u for u in fractions]
        # Keep a point-domain diagnostic attribute, useful when reviewing curls.
        attribute = surface.attributes.new("hair_rig_u", 'FLOAT', 'POINT')
        attribute.data.foreach_set("value", fractions)
        return surface, fractions, len(points)
    except Exception:
        if surface is not None:
            bpy.data.meshes.remove(surface)
        raise
    finally:
        if wire is not None:
            bpy.data.meshes.remove(wire)
        bpy.data.objects.remove(temporary, do_unlink=True)
        bpy.data.curves.remove(data)


def _activate(context, obj):
    for selected in context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def _next_version():
    versions = [int(collection[TAG]) for collection in bpy.data.collections if TAG in collection]
    versions += [int(obj[TAG]) for obj in bpy.data.objects if TAG in obj]
    versions += [int(bone[TAG]) for arm in bpy.data.armatures for bone in arm.bones if TAG in bone]
    versions += [int(collection[TAG]) for arm in bpy.data.armatures
                 for collection in arm.collections if TAG in collection]
    return max(versions, default=0) + 1


def _armature_collection(context, armature):
    """Choose one visible, editable owner in this view layer, preferring active."""
    owners = set(armature.users_collection)
    available = []

    def visit(layer):
        collection = layer.collection
        if (collection in owners and layer.visible_get()
                and collection.is_editable and not collection.hide_select):
            available.append(collection)
        # A scene root with no visible direct objects can report invisible while
        # its children remain visible. Each child evaluates its own ancestors.
        for child in layer.children:
            visit(child)

    visit(context.view_layer.layer_collection)
    active = context.view_layer.active_layer_collection.collection
    if active in available:
        return active
    if available:
        return available[0]
    raise RigError("Show an editable collection containing the destination armature first.")


def _next_armature_name():
    # Objects and armature data have separate namespaces. Reserve a name free in
    # both so Blender does not add a .001 suffix to either half of the new rig.
    pattern = re.compile(r"hair_armature_(\d+)(?:\.\d+)?$")
    numbers = [int(match.group(1))
               for block in (*bpy.data.objects, *bpy.data.armatures)
               if (match := pattern.fullmatch(block.name))]
    return f"hair_armature_{max(numbers, default=0) + 1:02d}"


def _strand_bone_name(lock, suffix):
    # Distinguish splines by their source index, independent of load/selection
    # order. Keep the original curve name, including spaces and Unicode.
    spline_suffix = f"_S{lock.spline_index + 1:02d}" if len(lock.source.data.splines) > 1 else ""
    ending = spline_suffix + suffix
    # Leave space for Blender's normal .001 collision suffix on 63-byte names.
    source_name = lock.source.name.encode('utf-8')[:59 - len(ending.encode('utf-8'))].decode('utf-8', errors='ignore')
    return source_name + ending


def generate(context, settings):
    locks, armature = prepare(context, settings)
    previous_selected = list(context.selected_objects)
    previous_active = context.view_layer.objects.active
    version = _next_version()
    label = f"HairRig_v{version:03d}"
    created_armature = armature is None
    if created_armature:
        collection = bpy.data.collections.new(label)
        context.scene.collection.children.link(collection)
        collection[TAG] = version
    else:
        collection = _armature_collection(context, armature)
    created_armature_data = None
    created_objects = []
    created_names = []
    objects = []
    staged_meshes = []
    bone_collection = None
    warnings = []
    try:
        # Finish every conversion before editing an existing armature.
        for lock in locks:
            try:
                mesh, fractions, ring_count = convert_lock(context, lock, collection)
            except RigError as error:
                raise RigError(f"{lock.label}: {error}") from error
            staged_meshes.append(mesh)
            objects.append((lock, mesh, fractions))
            if lock.skip_root_bones < lock.count and ring_count < 2 * lock.count + 1:
                warnings.append(f"{lock.label}: raise curve Resolution U for smoother bending")
        if created_armature:
            armature_name = _next_armature_name()
            data = bpy.data.armatures.new(armature_name)
            created_armature_data = data
            armature = bpy.data.objects.new(armature_name, data)
            created_objects.append(armature)
            collection.objects.link(armature)
            armature.show_in_front = True
            armature[TAG] = version
        inverse = armature.matrix_world.inverted()
        bone_collection = armature.data.collections.new(label)
        bone_collection[TAG] = version
        _activate(context, armature)
        bpy.ops.object.mode_set(mode='EDIT')
        edits = armature.data.edit_bones
        needs_anchor = any(lock.skip_root_bones for lock in locks)
        if created_armature:
            root = edits.new("Root")
            created_names.append(root.name)
            root.head = sum((inverse @ lock.joints[0] for lock in locks), Vector()) / len(locks)
            root.tail = root.head + Vector((0, 0, max((locks[0].joints[-1] - locks[0].joints[0]).length * .05, .01)))
            root.use_deform = needs_anchor
            root_name = root.name
        else:
            root_name = settings.parent_bone
        anchor_name = root_name
        if needs_anchor and not edits[root_name].use_deform:
            # A rig control may not deform meshes. Add one shared attachment
            # bone rather than changing any existing character bone's flags.
            parent = edits[root_name]
            anchor = edits.new("Anchor")
            created_names.append(anchor.name)
            anchor.head = parent.head
            anchor.tail = parent.tail
            anchor.roll = parent.roll
            anchor.parent = parent
            anchor.use_deform = True
            anchor_name = anchor.name
        chains = []
        for lock in locks:
            joints = [inverse @ p for p in lock.joints]
            normals = transported_normals(joints)
            previous = edits[root_name]
            names = []
            for i, (head, tail, normal) in enumerate(zip(joints, joints[1:], normals)):
                if i < lock.skip_root_bones:
                    continue
                bone = edits.new(_strand_bone_name(lock, f"_{i + 1:03d}"))
                created_names.append(bone.name)
                bone.head, bone.tail = head, tail
                bone.align_roll(normal)
                bone.parent = previous
                bone.use_connect = i > lock.skip_root_bones
                bone.use_deform = True
                names.append(bone.name)
                previous = bone
            if names:
                endpoint = edits.new(_strand_bone_name(lock, "_End"))
                created_names.append(endpoint.name)
                endpoint.head = joints[-1]
                direction = (joints[-1] - joints[-2])
                endpoint.tail = endpoint.head + direction.normalized() * max(direction.length * .1, 1e-5)
                endpoint.parent = previous
                endpoint.use_connect = True
                endpoint.use_deform = False
                endpoint.align_roll(normals[-1])
            chains.append(names)
        bpy.ops.object.mode_set(mode='OBJECT')
        for name in created_names:
            bone = armature.data.bones[name]
            bone[TAG] = version
            bone_collection.assign(bone)
            # New edit bones can inherit the active collection; keep versions isolated.
            for owner in list(bone.collections):
                if owner != bone_collection:
                    owner.unassign(bone)
        output = []
        for (lock, mesh, fractions), names in zip(objects, chains):
            name = f"{label}_{lock.source.name[:30]}_S{lock.spline_index + 1:02d}"
            mesh.name = name
            # Bake the complete transform into armature-local mesh coordinates.
            # This also preserves shear inherited from a transformed source parent,
            # which an Object's location/rotation/scale alone cannot represent.
            transform = inverse @ lock.source.matrix_world
            mesh.transform(transform)
            if transform.determinant() < 0:
                mesh.flip_normals()
            obj = bpy.data.objects.new(name, mesh)
            created_objects.append(obj)
            collection.objects.link(obj)
            obj.parent = armature
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix.Identity(4)
            obj[TAG] = version
            obj["hair_rig_source"] = lock.source.name
            obj["hair_rig_spline"] = lock.spline_index
            obj["hair_rig_root_bone"] = names[0] if names else ""
            obj["hair_rig_bone_count"] = len(names)
            obj["hair_rig_total_segments"] = lock.count
            obj["hair_rig_skip_root_bones"] = lock.skip_root_bones
            obj["hair_rig_anchor_bone"] = anchor_name if lock.skip_root_bones else ""
            groups = {i: obj.vertex_groups.new(name=name)
                      for i, name in enumerate(names, start=lock.skip_root_bones)}
            if lock.skip_root_bones:
                groups[-1] = obj.vertex_groups.new(name=anchor_name)
            for vertex_index, fraction in enumerate(fractions):
                for bone_index, weight in weights_at(fraction, lock.count, lock.skip_root_bones):
                    groups[bone_index].add([vertex_index], weight, 'REPLACE')
            modifier = obj.modifiers.new("Hair Rig", 'ARMATURE')
            modifier.object = armature
            modifier.use_deform_preserve_volume = False  # Match ordinary engine skinning.
            modifier.use_vertex_groups = True
            modifier.use_bone_envelopes = False
            output.append(obj)
        if created_armature:
            collection["hair_rig_armature"] = armature.name
            collection["hair_rig_bone_collection"] = bone_collection.name
        _activate(context, armature)
        for obj in output:
            obj.select_set(True)
        # The scene's root collection is embedded data and cannot be stored in
        # a PointerProperty. Version/armature tracking also covers that case.
        settings.latest_collection = None if collection.is_embedded_data else collection
        settings.latest_armature = armature
        settings.latest_version = version
        moving_count = sum(lock.count - lock.skip_root_bones for lock in locks)
        settings.status = f"Created {label}: {len(locks)} locks, {moving_count} moving hair bones."
        return collection, armature, output, warnings
    except Exception:
        # Roll back only this generation, including any partially created bones.
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if armature is not None and not created_armature and created_names:
            _activate(context, armature)
            bpy.ops.object.mode_set(mode='EDIT')
            for name in reversed(created_names):
                bone = armature.data.edit_bones.get(name)
                if bone:
                    armature.data.edit_bones.remove(bone)
            bpy.ops.object.mode_set(mode='OBJECT')
        if bone_collection is not None and not created_armature:
            armature.data.collections.remove(bone_collection)
        for obj in reversed(created_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        if created_armature_data is not None and created_armature_data.users == 0:
            bpy.data.armatures.remove(created_armature_data)
        for mesh in staged_meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        if created_armature:
            bpy.data.collections.remove(collection)
        for selected in context.selected_objects:
            selected.select_set(False)
        for obj in previous_selected:
            obj.select_set(True)
        context.view_layer.objects.active = previous_active
        raise
