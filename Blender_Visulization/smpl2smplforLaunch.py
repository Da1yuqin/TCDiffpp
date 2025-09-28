import bpy
import os
import sys


SMPL_DIR = str(sys.argv[-5])
OUT_DIR = str(sys.argv[-3])
MAIN_DIR = str(sys.argv[-2])


def purge_orphans():
    if bpy.app.version >= (3, 0, 0):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    else:
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()
            
def get_total_animation_frames(obj_name):
    active_object=bpy.data.objects.get(obj_name)
    print(active_object)
    animation_data=active_object.animation_data
    print(animation_data)
    action=animation_data.action
    print(action)
    print(action.frame_range)
    total_frames=int(action.frame_range[1])
    return total_frames

print(sys.argv)
count = int(sys.argv[-1])
file = os.path.splitext(os.path.basename(sys.argv[-4]))[0] + ".fbx"
print("begin:{0}".format(count))

# Clear the current scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX and Skinning
ffp = SMPL_DIR
bpy.ops.import_scene.fbx(filepath=ffp)
mfp = MAIN_DIR+"\\"+file
bpy.ops.import_scene.fbx(filepath=mfp)
scene = bpy.context.scene

## retarget
target_obj = "smpl"
objname = "smpl.001"
for obj in scene.objects:
    if obj.name == target_obj:
        scene.target_rig = obj.name
    if obj.name == objname:
        scene.source_rig = obj.name
    
bpy.ops.arp.auto_scale()
bpy.ops.arp.build_bones_list()
bpy.ops.arp.import_config(filepath="src\\smpl2smpl.bmap")
frame_len=get_total_animation_frames(objname)
print("frame_len:{0}".format(frame_len))
bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=0, frame_end=frame_len)
    
scene.frame_start = 0
scene.frame_end = frame_len
bpy.ops.object.select_all(action='DESELECT')
objects_to_color = [
    bpy.data.objects['Alpha_Surface.035'],
    bpy.data.objects['Alpha_Surface.034'],
    bpy.data.objects['Alpha_Surface.015'],
    bpy.data.objects['Alpha_Surface.014'],
    bpy.data.objects['Alpha_Surface.013'],
    bpy.data.objects['Alpha_Surface.016'],
    bpy.data.objects['Alpha_Surface.017'],
    bpy.data.objects['Alpha_Surface.012'],
    bpy.data.objects['Alpha_Surface.011'],
    bpy.data.objects['Alpha_Surface.010'],
    bpy.data.objects['Alpha_Surface.009'],
    bpy.data.objects['Alpha_Surface.008'],
    bpy.data.objects['Alpha_Surface.007'],
    bpy.data.objects['Alpha_Surface.006'],
    bpy.data.objects['Alpha_Surface.005'],
    bpy.data.objects['Alpha_Surface.003'],
    bpy.data.objects['Alpha_Surface.007'],
    # fingers
    bpy.data.objects['Alpha_Surface.036'],
    bpy.data.objects['Alpha_Surface.037'],
    bpy.data.objects['Alpha_Surface.038'],
    bpy.data.objects['Alpha_Surface.039'],
    bpy.data.objects['Alpha_Surface.040'],
    bpy.data.objects['Alpha_Surface.041'],
    bpy.data.objects['Alpha_Surface.042'],
    bpy.data.objects['Alpha_Surface.043'],
    bpy.data.objects['Alpha_Surface.044'],
    bpy.data.objects['Alpha_Surface.045'],
    bpy.data.objects['Alpha_Surface.046'],
    bpy.data.objects['Alpha_Surface.047'],
    bpy.data.objects['Alpha_Surface.048'],
    bpy.data.objects['Alpha_Surface.049'],
    bpy.data.objects['Alpha_Surface.050'],
    bpy.data.objects['Alpha_Surface.051'],
    # lefthand
    bpy.data.objects['Alpha_Surface.018'],
    bpy.data.objects['Alpha_Surface.019'],
    bpy.data.objects['Alpha_Surface.020'],
    bpy.data.objects['Alpha_Surface.021'],
    bpy.data.objects['Alpha_Surface.022'],
    bpy.data.objects['Alpha_Surface.023'],
    bpy.data.objects['Alpha_Surface.024'],
    bpy.data.objects['Alpha_Surface.025'],
    bpy.data.objects['Alpha_Surface.026'],
    bpy.data.objects['Alpha_Surface.027'],
    bpy.data.objects['Alpha_Surface.028'],
    bpy.data.objects['Alpha_Surface.029'],
    bpy.data.objects['Alpha_Surface.030'],
    bpy.data.objects['Alpha_Surface.031'],
    bpy.data.objects['Alpha_Surface.032'],
    bpy.data.objects['Alpha_Surface.033'], 
]

mat = bpy.data.materials.new(name="MyMaterial")
if count % 5 == 0: 
    mat.diffuse_color = (0.855, 0.047, 0.032, 1.0)   # red EE3D32
elif count % 5 == 1:
    mat.diffuse_color = (0.156, 0.863, 1.0, 1.0)  # blue 6EEFFF
elif count % 5 == 2:
    mat.diffuse_color = (0.651, 0.1, 1.0, 1.0)  # purple D359FF
elif count % 5 == 3:
    mat.diffuse_color = (1.0, 0.597, 0.0, 1.0)  # yellow FFCB00
elif count % 5 == 4:
    mat.diffuse_color = (0.026, 0.651, 0.386, 1.0)   # green 2DD3A7


bpy.ops.object.select_all(action='DESELECT')
for obj in objects_to_color:
    obj.data.materials.append(mat)
    obj.active_material = mat     


objects_to_select = [
    bpy.data.objects['smpl'],
    bpy.data.objects['Alpha_Surface.003'],
    bpy.data.objects['Alpha_Surface.011'],
    bpy.data.objects['Alpha_Surface.036'],
    bpy.data.objects['Alpha_Surface.045'],
    bpy.data.objects['Alpha_Joints.024'],
    bpy.data.objects['Alpha_Surface.006'],
    bpy.data.objects['Alpha_Surface.012'],
    bpy.data.objects['Alpha_Surface.049'],
    bpy.data.objects['Alpha_Joints.021'],
    bpy.data.objects['Alpha_Joints.037'],
    bpy.data.objects['Alpha_Surface.005'],
    bpy.data.objects['Alpha_Surface.050'],
    bpy.data.objects['Alpha_Joints.020'],
    bpy.data.objects['Alpha_Joints.041'],
    bpy.data.objects['Alpha_Surface.016'],
    bpy.data.objects['Alpha_Surface.030'],
    bpy.data.objects['Alpha_Surface.046'],
    bpy.data.objects['Alpha_Joints.019'],
    bpy.data.objects['Alpha_Joints.038'],
    bpy.data.objects['Alpha_Surface.013'],
    bpy.data.objects['Alpha_Surface.028'],
    bpy.data.objects['Alpha_Joints.006'],
    bpy.data.objects['Alpha_Joints.018'],
    bpy.data.objects['Alpha_Joints.047'],
    bpy.data.objects['Alpha_Surface.018'],
    bpy.data.objects['Alpha_Joints.005'],
    bpy.data.objects['Alpha_Joints.017'],
    bpy.data.objects['Alpha_Joints.032'],
    bpy.data.objects['Alpha_Joints.045'],
    bpy.data.objects['Alpha_Joints.049'],
    bpy.data.objects['Alpha_Joints.022'],
    bpy.data.objects['Alpha_Joints.030'],
    bpy.data.objects['Alpha_Surface.017'],
    bpy.data.objects['Alpha_Surface.019'],
    bpy.data.objects['Alpha_Surface.029'],
    bpy.data.objects['Alpha_Surface.043'],
    bpy.data.objects['Alpha_Joints.007'],
    bpy.data.objects['Alpha_Surface.015'],
    bpy.data.objects['Alpha_Surface.033'],
    bpy.data.objects['Alpha_Surface.040'],
    bpy.data.objects['Alpha_Joints.012'],
    bpy.data.objects['Alpha_Joints.016'],
    bpy.data.objects['Alpha_Joints.035'],
    bpy.data.objects['Alpha_Joints.040'],
    bpy.data.objects['Alpha_Surface.020'],
    bpy.data.objects['Alpha_Surface.038'],
    bpy.data.objects['Alpha_Joints.009'],
    bpy.data.objects['Alpha_Joints.028'],
    bpy.data.objects['Alpha_Joints.043'],
    bpy.data.objects['Alpha_Surface.024'],
    bpy.data.objects['Alpha_Surface.037'],
    bpy.data.objects['Alpha_Surface.039'],
    bpy.data.objects['Alpha_Surface.041'],
    bpy.data.objects['Alpha_Surface.048'],
    bpy.data.objects['Alpha_Joints.011'],
    bpy.data.objects['Alpha_Joints.027'],
    bpy.data.objects['Alpha_Joints.036'],
    bpy.data.objects['Alpha_Surface.014'],
    bpy.data.objects['Alpha_Surface.021'],
    bpy.data.objects['Alpha_Surface.023'],
    bpy.data.objects['Alpha_Surface.047'],
    bpy.data.objects['Alpha_Joints.004'],
    bpy.data.objects['Alpha_Joints.026'],
    bpy.data.objects['Alpha_Surface.022'],
    bpy.data.objects['Alpha_Surface.027'],
    bpy.data.objects['Alpha_Joints.008'],
    bpy.data.objects['Alpha_Joints.013'],
    bpy.data.objects['Alpha_Joints.039'],
    bpy.data.objects['Alpha_Surface.007'],
    bpy.data.objects['Alpha_Surface.026'],
    bpy.data.objects['Alpha_Surface.032'],
    bpy.data.objects['Alpha_Surface.051'],
    bpy.data.objects['Alpha_Joints.031'],
    bpy.data.objects['Alpha_Joints.044'],
    bpy.data.objects['Alpha_Joints.046'],
    bpy.data.objects['Alpha_Surface.008'],
    bpy.data.objects['Alpha_Surface.025'],
    bpy.data.objects['Alpha_Surface.035'],
    bpy.data.objects['Alpha_Surface.042'],
    bpy.data.objects['Alpha_Joints.025'],
    bpy.data.objects['Alpha_Joints.033'],
    bpy.data.objects['Alpha_Joints.042'],
    bpy.data.objects['Alpha_Joints.048'],
    bpy.data.objects['Alpha_Joints.015'],
    bpy.data.objects['Alpha_Joints.029'],
    bpy.data.objects['Alpha_Surface.034'],
    bpy.data.objects['Alpha_Joints.003'],
    bpy.data.objects['Alpha_Surface.009'],
    bpy.data.objects['Alpha_Surface.031'],
    bpy.data.objects['Alpha_Joints.010'],
    bpy.data.objects['Alpha_Joints.034'],
    bpy.data.objects['Alpha_Surface.010'],
    bpy.data.objects['Alpha_Surface.044'],
    bpy.data.objects['Alpha_Joints.014'],
    bpy.data.objects['Alpha_Joints.023']
]


bpy.ops.object.select_all(action='DESELECT')
for obj in objects_to_select:
    obj.select_set(True)

new_name = OUT_DIR +"\\" + file[:-4] + ".fbx" 

bpy.ops.export_scene.fbx(
    filepath=new_name,
    use_selection=True,
    apply_scale_options='FBX_SCALE_ALL',
    axis_forward='-Z',
    axis_up='Y',
    global_scale=1.0,
    bake_space_transform=True,
    object_types={'EMPTY', 'MESH', 'ARMATURE', 'CAMERA', 'LIGHT'},
    use_mesh_modifiers=True,
    mesh_smooth_type='OFF',
    use_armature_deform_only=True,
    add_leaf_bones=False,
    bake_anim=True,
    bake_anim_use_nla_strips=True,
    bake_anim_use_all_actions=True
)
