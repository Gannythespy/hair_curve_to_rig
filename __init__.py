"""Hair Curve Rig — ordinary bones and skin weights from beveled Bézier curves."""

bl_info = {
    "name": "Hair Curve Rig",
    "author": "Gani",
    "version": (1, 2, 1),
    "blender": (5, 3, 0),
    "location": "3D Viewport > Sidebar > Hair Rig",
    "description": "Generate weighted hair chains with fixed roots in new or existing character rigs",
    "category": "Rigging",
}

import traceback

# Blender reloads the package when an installed add-on is updated. Refresh its
# implementation modules too, so an upgrade does not retain 1.0's generator.
if "bpy" in locals():
    from importlib import reload
    from . import geometry, rigging
    reload(geometry)
    reload(rigging)

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup, UIList

from .geometry import RigError
from .rigging import TAG, generate, validate_source


def armature_poll(self, obj):
    return obj.type == 'ARMATURE'


def clamp_skipped_roots(self, context):
    if self.skip_root_bones > self.bone_count:
        self.skip_root_bones = self.bone_count


class HAIRRIG_PG_lock(PropertyGroup):
    source: PointerProperty(type=bpy.types.Object)
    spline_index: IntProperty(min=0)
    point_count: IntProperty(min=0)
    bone_count: IntProperty(
        name="Bones", default=5, min=1, soft_max=32, max=1024,
        description="Number of segments along the full path, before skipping root bones",
        update=clamp_skipped_roots,
    )
    skip_root_bones: IntProperty(
        name="Skip Root Bones", default=0, min=0, soft_max=10, max=1024,
        description="Omit this many consecutive bones from the strand root; their section follows the head or shared root",
        update=clamp_skipped_roots,
    )
    enabled: BoolProperty(name="Include", default=True)
    reverse: BoolProperty(
        name="Reverse Root/Tip", default=False,
        description="Use the spline's last control point as the root; the source curve is unchanged",
    )


class HAIRRIG_PG_settings(PropertyGroup):
    locks: CollectionProperty(type=HAIRRIG_PG_lock)
    active_lock: IntProperty(default=0)
    destination: EnumProperty(
        name="Destination",
        items=[('NEW', "New Hair Armature", "Create an armature with a shared hair root"),
               ('EXISTING', "Existing Armature", "Add chains under a chosen character bone")],
        default='NEW',
    )
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=armature_poll)
    parent_bone: StringProperty(name="Parent Bone", description="Usually the character's head bone")
    latest_collection: PointerProperty(type=bpy.types.Collection)
    latest_armature: PointerProperty(type=bpy.types.Object)
    latest_version: IntProperty(default=0, min=0)
    status: StringProperty(default="Select beveled Bézier hair curves, then load them.")


class HAIRRIG_OT_load(Operator):
    bl_idname = "hair_rig.load_curves"
    bl_label = "Load Selected Curves"
    bl_description = "List each selected hair spline; bone counts start at its control-point count"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.hair_curve_rig
        curves = [obj for obj in context.selected_objects if obj.type == 'CURVE']
        profiles = {obj.data.bevel_object for obj in curves} | {obj.data.taper_object for obj in curves}
        curves = sorted((obj for obj in curves if obj not in profiles), key=lambda obj: obj.name)
        if not curves:
            self.report({'ERROR'}, "Select at least one beveled Bézier hair curve.")
            return {'CANCELLED'}
        rows = []
        try:
            for obj in curves:
                if not obj.data.splines:
                    raise RigError(f"{obj.name}: the curve has no splines.")
                for index in range(len(obj.data.splines)):
                    spline = validate_source(obj, index)
                    rows.append((obj, index, len(spline.bezier_points)))
        except RigError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        existing = {
            (row.source.as_pointer(), row.spline_index, row.point_count):
                (row.bone_count, row.reverse, row.enabled, row.skip_root_bones)
            for row in settings.locks if row.source
        }
        settings.locks.clear()
        for obj, index, points in rows:
            row = settings.locks.add()
            row.source, row.spline_index, row.point_count = obj, index, points
            row.bone_count, row.reverse, row.enabled, row.skip_root_bones = existing.get(
                (obj.as_pointer(), index, points), (points, False, True, 0),
            )
        settings.active_lock = 0
        settings.status = f"Loaded {len(rows)} hair splines. Adjust counts, then generate."
        return {'FINISHED'}


class HAIRRIG_OT_generate(Operator):
    bl_idname = "hair_rig.generate"
    bl_label = "Generate New Version"
    bl_description = "Create mesh copies, chains, and weights; preserve all previous versions"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and bool(context.scene.hair_curve_rig.locks)

    def execute(self, context):
        settings = context.scene.hair_curve_rig
        try:
            _, _, _, warnings = generate(context, settings)
        except RigError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        except Exception as error:
            traceback.print_exc()
            self.report({'ERROR'}, f"Generation rolled back: {error}")
            return {'CANCELLED'}
        self.report({'INFO'}, settings.status)
        if warnings:
            self.report({'WARNING'}, f"{len(warnings)} lock(s) have few surface rings. Increase source Resolution U for smoother bends.")
        return {'FINISHED'}


class HAIRRIG_OT_select(Operator):
    bl_idname = "hair_rig.select"
    bl_label = "Select Hair Objects"
    bl_description = "Select source curves or the latest generated meshes and armature"
    bl_options = {'REGISTER', 'UNDO'}
    target: EnumProperty(items=[('SOURCE', "Sources", ""), ('OUTPUT', "Latest Output", "")])

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        settings = context.scene.hair_curve_rig
        if self.target == 'SOURCE':
            objects = list({row.source for row in settings.locks if row.source})
        elif settings.latest_version:
            objects = [obj for obj in context.scene.objects
                       if obj.type == 'MESH' and obj.get(TAG) == settings.latest_version]
            if settings.latest_armature:
                objects.append(settings.latest_armature)
        else:
            # Files saved before 1.2.1 track output through a dedicated collection.
            collection = settings.latest_collection
            if not collection:
                self.report({'ERROR'}, "No generated output is available.")
                return {'CANCELLED'}
            objects = list(collection.objects)
            armature = bpy.data.objects.get(collection.get("hair_rig_armature", ""))
            if armature and armature not in objects:
                objects.append(armature)
        available = [obj for obj in objects if obj.name in context.view_layer.objects
                     and obj.visible_get() and not obj.hide_select]
        if not available:
            self.report({'ERROR'}, "Show these objects in the Outliner before selecting them.")
            return {'CANCELLED'}
        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in available:
            obj.select_set(True)
        context.view_layer.objects.active = next(
            (obj for obj in available if obj.type == 'ARMATURE'), available[0],
        )
        return {'FINISHED'}


class HAIRRIG_UL_locks(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        label = f"{item.source.name} · {item.spline_index + 1}" if item.source else "Missing curve"
        row.label(text=label, icon='CURVE_BEZCURVE')
        row.prop(item, "bone_count", text="")


class HAIRRIG_PT_main(Panel):
    bl_label = "Hair Curve Rig"
    bl_idname = "HAIRRIG_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hair Rig"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.hair_curve_rig
        layout.operator("hair_rig.load_curves", icon='CURVE_BEZCURVE')
        if settings.locks:
            layout.template_list("HAIRRIG_UL_locks", "", settings, "locks", settings, "active_lock", rows=4)
            if settings.active_lock < len(settings.locks):
                item = settings.locks[settings.active_lock]
                box = layout.box()
                box.label(text=f"{item.point_count} control points · {item.bone_count} total segments")
                box.prop(item, "bone_count")
                box.prop(item, "skip_root_bones")
                moving = item.bone_count - item.skip_root_bones
                if item.skip_root_bones:
                    box.label(text=f"{item.skip_root_bones} skipped · {moving} moving bones")
                    if moving == 0:
                        box.label(text="Entire lock follows the head / shared root.")
                box.prop(item, "reverse")
            layout.label(text="Skipped sections stay attached to the head.", icon='INFO')
        layout.prop(settings, "destination", text="")
        if settings.destination == 'EXISTING':
            layout.prop(settings, "armature")
            if settings.armature:
                layout.prop_search(settings, "parent_bone", settings.armature.data, "bones")
        layout.separator()
        layout.operator("hair_rig.generate", icon='ARMATURE_DATA')
        layout.label(text="Each generation keeps earlier versions.")
        if settings.latest_collection or settings.latest_version:
            if settings.latest_version:
                layout.label(text=f"HairRig_v{settings.latest_version:03d}")
            if settings.latest_collection:
                layout.label(text=settings.latest_collection.name, icon='OUTLINER_COLLECTION')
            row = layout.row(align=True)
            row.operator("hair_rig.select", text="Select Sources").target = 'SOURCE'
            row.operator("hair_rig.select", text="Select Output").target = 'OUTPUT'
            layout.label(text="Hide sources / old versions in the Outliner.")
        if context.mode != 'OBJECT':
            layout.label(text="Switch to Object Mode to generate.", icon='INFO')


CLASSES = (
    HAIRRIG_PG_lock, HAIRRIG_PG_settings, HAIRRIG_OT_load,
    HAIRRIG_OT_generate, HAIRRIG_OT_select, HAIRRIG_UL_locks, HAIRRIG_PT_main,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hair_curve_rig = PointerProperty(type=HAIRRIG_PG_settings)


def unregister():
    del bpy.types.Scene.hair_curve_rig
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
