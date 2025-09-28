import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)
from tqdm import tqdm
from dataset.quaternion import ax_from_6v, quat_slerp
from p_tqdm import p_map
import pickle

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

# Parent indices for SMPL joints
smpl_parents = [
    -1, # root
    0,  # lhip
    0,  # rhip
    0,  # belly
    1,  # lknee
    2,  # rknee
    3,  # spine
    4,  # lankle
    5,  # rankle
    6,  # chest
    7,  # ltoes
    8,  # rtoes
    9,  # neck
    9,  # linshoulder
    9,  # rinshoulder
    12, # head
    13, # lshoulder
    14, # rshoulder
    16, # lelbow
    17, # relbow
    18, # lwrist
    19, # rwrist
    20, # lhand
    21, # rhand
]

# Offsets for SMPL joints
smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    """Set 3D line data for plotting."""
    if len(x.shape) == 2:  # Single person (2, 3)
        line.set_data(x[:, :2].T)
        line.set_3d_properties(x[:, 2])
    else:  # Multi-dancer: (dancer_num, 2, 3)
        for i in range(x.shape[0]):
            line[i].set_data(x[i, :, :2].T)
            line[i].set_3d_properties(x[i, :, 2])


def set_scatter_data_3d(scat, x, c):
    """Set 3D scatter data for plotting."""
    if len(x.shape) == 3:  # Multi-dancer: (dancer_num, 1, 3)
        dancer_num = x.shape[0]
        for i in range(dancer_num):
            scat[i].set_offsets(x[i, :, :2])
            scat[i].set_3d_properties(x[i, :, 2], "z")
            scat[i].set_facecolors([c[i]])
    else:  # Single person
        scat.set_offsets(x[:, :2])
        scat.set_3d_properties(x[:, 2], "z")
        scat.set_facecolors([c])


def get_axrange(poses):
    """Calculate the axis range for plotting based on pose data."""
    pose = poses[0]
    x_min, x_max = pose[:, 0].min(), pose[:, 0].max()
    y_min, y_max = pose[:, 1].min(), pose[:, 1].max()
    z_min, z_max = pose[:, 2].min(), pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    return max([xdiff, ydiff, zdiff])


def plot_multi_pose(num, poses, lines, ax, axrange, scat, contact):
    """Plot multi-dancer poses in 3D."""
    pose = poses[:, num]  # (J, 3) -> (n, J, 3), n=1...5
    static = contact[:, num]  # (n, 4), n=1...5
    dancer_num = poses.shape[0]  # (dancer_num, 150, 24, 3)
    indices = [7, 8, 10, 11]  # Foot joint indices

    # Plot points
    for i, idx in enumerate(indices):
        position = pose[:, idx : idx + 1]  # (1, 3) -> (dancer_num, 1, 3)
        color = ["r" if static[dancer_i, i] else "g" for dancer_i in range(dancer_num)]

        for point_id in range(dancer_num):
            scat[point_id * 4 + i].set_offsets(position[point_id, :, :2])
            scat[point_id * 4 + i].set_3d_properties(position[point_id, :, 2], "z")
            scat[point_id * 4 + i].set_facecolors([color[point_id]])

    # Plot lines
    for i, p in enumerate(smpl_parents):
        if i == 0:  # Skip root joint
            if num > 1:  # Draw trajectory for root joint
                for line_id in range(dancer_num):
                    lines[line_id][0].set_data(poses[line_id, :num, 0, :2].T)
                    lines[line_id][0].set_3d_properties(0)
            continue

        data = np.stack((pose[:, i], pose[:, p]), axis=1)  # (2, 3) -> (dancer_num, 2, 3)
        for line_id in range(dancer_num):
            lines[line_id][i].set_data(data[line_id, :, :2].T)
            lines[line_id][i].set_3d_properties(data[line_id, :, 2])

    # Set axis limits
    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        ax.set_xlim(xcenter - stepx, xcenter + stepx)
        ax.set_ylim(ycenter - stepy, ycenter + stepy)
        ax.set_zlim(zcenter - stepz, zcenter + stepz)

    return lines, scat


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact):
    """Plot single dancer pose in 3D."""
    pose = poses[num]  # (J, 3)
    static = contact[num]
    indices = [7, 8, 10, 11]  # Foot joint indices

    # Plot points
    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]  # (1, 3)
        color = "r" if static[i] else "g"  # Red if foot is static, else green
        set_scatter_data_3d(point, position, color)

    # Plot lines
    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        if i == 0:  # Skip root joint
            if num > 1:  # Draw trajectory for root joint
                lines[0].set_data(poses[:num, 0, :2].T)
                lines[0].set_3d_properties(0)
            continue

        data = np.stack((pose[i], pose[p]), axis=0)  # (2, 3)
        set_line_data_3d(line, data)

    # Set axis limits
    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 2.5, 3.5, 2.5  
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        ax.set_xlim(xcenter - stepx, xcenter + stepx)
        ax.set_ylim(ycenter - stepy, ycenter + stepy)
        ax.set_zlim(zcenter - stepz, zcenter + stepz)


def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True,
    interaction_list=None,
):
    """Render skeleton animations with optional audio."""
    dancer_num = 1
    multi_dancer = False
    if len(poses.shape) == 4:  # Multi-dancer
        multi_dancer = True
        dancer_num = poses.shape[0]

    if render:
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[1] if multi_dancer else poses.shape[0]  # 150

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=40, azim=-90)  # Set camera angle

        # Create lines and scatter points
        colors = ["#e3ba8f", "#ff6b6b", "#0abde3", "#576574", "#01a3a4"]
        lines = [
            [
                ax.plot([], [], [], zorder=10, linewidth=4.0, color=colors[dancer_i])[0]
                for _ in smpl_parents
            ]
            for dancer_i in range(dancer_num)
        ]
        scat = [
            ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            for _ in range(4 * dancer_num)
        ]

        axrange = 4  # Plot range

        # Compute contact labels
        if multi_dancer:
            feet = poses[:, :, (7, 8, 10, 11)]
            feetv = np.zeros(feet.shape[:3])
        else:
            feet = poses[:, (7, 8, 10, 11)]
            feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        contact = contact if contact is not None else feetv < 0.01

        # Create animation
        plot_pose = plot_multi_pose if multi_dancer else plot_single_pose
        anim = animation.FuncAnimation(
            fig,
            func=plot_pose,
            frames=num_steps,
            fargs=(poses, lines, ax, axrange, scat, contact),
            interval=1000 // 30,
        )

        fig.tight_layout()

    if sound:
        # Save animation with audio
        if render:
            temp_dir = TemporaryDirectory()
            gifname = os.path.join(temp_dir.name, f"{epoch}.gif")
            anim.save(gifname)

        if stitch:
            assert isinstance(name, list)
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4')
        else:
            assert isinstance(name, str) and name != ""
            audioname = name
            outname = os.path.join(out, f"{epoch}_{os.path.splitext(os.path.basename(name))[0]}.mp4")

        if render:
            os.system(f"/usr/bin/ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -c:a libmp3lame -q:a 4 {outname}")
    elif render:
        path = os.path.normpath(name)
        pathparts = path.split(os.sep)
        gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
        anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"})

    plt.close()


class SMPLSkeleton:
    def __init__(self, device=None):
        """
        Initialize the SMPL Skeleton model.
        :param device: The device to use for computations (e.g., 'cpu' or 'cuda').
        """
        offsets = smpl_offsets  # Assuming these are defined elsewhere
        parents = smpl_parents  # Assuming these are defined elsewhere
        assert len(offsets) == len(parents), "Offsets and parents must have the same length."

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        """
        Compute metadata such as which joints have children and list of children for each joint.
        """
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = [[] for _ in range(len(self._parents))]
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        :param rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
        :param root_positions: (N, L, 3) tensor describing the root joint positions.
        :return: A tensor representing the world positions of all joints.
        """
        assert len(rotations.shape) == 4, "Rotations should be a 4D tensor."
        assert len(root_positions.shape) == 3, "Root positions should be a 3D tensor."

        # Convert from axis-angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],  # Batch size
            rotations.shape[1],  # Sequence length
            self._offsets.shape[0],  # Number of joints
            self._offsets.shape[1]   # Offset dimensions (usually 3)
        )

        # Iterate over each joint to compute its world position and rotation
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                # Root joint
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                # Non-root joint
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    ) + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    # If the joint has children, update its world rotation
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # Terminal node, no need to compute transformation
                    rotations_world.append(None)

        # Stack and permute dimensions to get final output shape
        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)


def render_dance_sample(
    samples,
    normalizer,
    epoch,
    render_out,
    fk_out=None,
    name=None,
    sound=True,
    mode="normal",
    render=True,
    required_dancer_num=3,
    x_0=None,
):
    """
    Render dance samples with skeletal visualization and optional audio.
    
    Args:
        samples: Input motion data tensor
        normalizer: Data normalization module
        epoch: Current epoch number for naming outputs
        render_out: Output directory for rendered animations
        fk_out: Directory to save forward kinematics data (optional)
        name: Audio filename(s) for synchronization
        sound: Enable audio rendering
        mode: Rendering mode ('normal' or 'long')
        render: Enable visualization rendering
        required_dancer_num: Number of dancers in the scene
        x_0: Initial state for rendering (unused in current implementation)
    """
    smpl = SMPLSkeleton(samples.device)

    # Normalize and reshape input data
    batch_size, seq_length, _ = samples.shape  # [*, 450, 151]
    normalized_samples = normalizer.unnormalize(samples)
    
    # Reshape to (batch_size, seq_length, dancer_num, features)
    reshaped_samples = normalized_samples.reshape(batch_size, 150, -1, 151)
    
    # Separate contact information from motion data
    if len(reshaped_samples.shape) == 4 and reshaped_samples.shape[3] == 151:
        sample_contact, motion_data = torch.split(reshaped_samples, [4, 147], dim=3)
    else:
        sample_contact = None

    # Prepare position and rotation data for FK
    batch_size, seq_length, dancer_num, _ = motion_data.shape
    positions = motion_data[..., :3].to(samples.device)
    rotations = ax_from_6v(motion_data[..., 3:].reshape(batch_size, -1, 24, 6)).to(samples.device)

    if mode == "long":
        # Process long sequences with temporal stitching
        self._process_long_sequence(
            positions,
            rotations,
            smpl,
            epoch,
            render_out,
            fk_out,
            name,
            sound,
            required_dancer_num,
            batch_size,
            seq_length,
            dancer_num
        )
        return

    # Standard processing for normal mode
    joint_positions = smpl.forward(rotations, positions).detach().cpu().numpy()
    
    # Prepare contact information
    if sample_contact is not None:
        sample_contact = np.transpose(sample_contact.detach().cpu().numpy(), (0, 2, 1, 3))

    # Reshape for visualization (batch_size, dancer_num, seq_length, joints, 3)
    vis_poses = joint_positions.reshape(batch_size, -1, required_dancer_num, 24, 3)
    vis_poses = np.transpose(vis_poses, (0, 2, 1, 3, 4))

    # Parallel rendering
    self._parallel_render(vis_poses, epoch, render_out, name, sound, sample_contact)

    # Save FK data if specified
    if fk_out:
        self._save_fk_data(rotations, positions, name, epoch, fk_out, vis_poses)

def _process_long_sequence(
    self,
    positions,
    rotations,
    smpl,
    epoch,
    render_out,
    fk_out,
    name,
    sound,
    dancer_num,
    batch_size,
    seq_length,
    original_dancer_num
):
    """Process long sequences with temporal stitching and interpolation."""
    # Reshape for sequence stitching
    positions = positions.reshape(batch_size, seq_length, dancer_num, 3)
    rotations = rotations.reshape(batch_size, seq_length, dancer_num, 24, 3)

    if batch_size > 1:
        # Temporal blending for multi-segment sequences
        full_positions, full_rotations = [], []
        
        for dancer_idx in range(dancer_num):
            # Position blending with linear interpolation
            blended_pos = self._blend_positions(
                positions[:, :, dancer_idx], 
                seq_length
            )
            
            # Rotation blending with spherical interpolation
            blended_rot = self._blend_rotations(
                rotations[:, :, dancer_idx],
                seq_length
            )
            
            full_positions.append(blended_pos)
            full_rotations.append(blended_rot)

        # Combine all dancers' data
        combined_pos = torch.cat(full_positions, dim=1)
        combined_rot = torch.cat(full_rotations, dim=1)
    else:
        combined_pos = positions
        combined_rot = rotations

    # Calculate final poses
    final_poses = smpl.forward(
        combined_rot.flatten(end_dim=1), 
        combined_pos.flatten(end_dim=1)
    ).detach().cpu().numpy()

    # Reshape for visualization (dancer_num, full_length, joints, 3)
    vis_poses = final_poses.reshape(
        -1, dancer_num, combined_pos.shape[1], 24, 3
    ).transpose(0, 2, 1, 3, 4)

    # Render and save results
    skeleton_render(
        vis_poses[0],
        epoch=str(epoch),
        out=render_out,
        name=name,
        sound=sound,
        stitch=True,
        render=render
    )

    # Save FK data if specified
    if fk_out:
        self._save_long_fk_data(
            combined_rot,
            combined_pos,
            name,
            epoch,
            fk_out,
            vis_poses
        )

def _blend_positions(self, positions, seq_length):
    """Blend positions with linear interpolation between segments."""
    half_length = seq_length // 2
    blend_weights = torch.linspace(1, 0, half_length, device=positions.device)
    
    # Apply fading to sequence segments
    positions[:-1] *= blend_weights.view(1, -1, 1)
    positions[1:] *= 1 - blend_weights.view(1, -1, 1)
    
    # Stitch segments together
    blended = torch.zeros(
        (seq_length + half_length * (positions.shape[0]-1), 3),
        device=positions.device
    )
    
    current_idx = 0
    for segment in positions:
        blended[current_idx:current_idx+seq_length] += segment
        current_idx += half_length
        
    return blended.unsqueeze(0)

def _blend_rotations(self, rotations, seq_length):
    """Blend rotations using spherical interpolation."""
    half_length = seq_length // 2
    quats = axis_angle_to_quaternion(rotations)
    
    # Perform slerp between segment boundaries
    blended_segments = []
    for i in range(rotations.shape[0]-1):
        start = quats[i, half_length:]
        end = quats[i+1, :half_length]
        
        interp_weights = torch.linspace(0, 1, half_length, device=quats.device)
        interp_quats = quat_slerp(start, end, interp_weights)
        
        blended_segments.append(interp_quats)
    
    # Combine all segments
    full_sequence = torch.cat([
        quats[0, :half_length],
        *blended_segments,
        quats[-1, half_length:]
    ])
    
    return quaternion_to_axis_angle(full_sequence).unsqueeze(0)

def _parallel_render(self, poses, epoch, render_out, name, sound, contacts):
    """Parallel rendering of poses using multiprocessing."""
    def render_worker(args):
        idx, pose = args
        skeleton_render(
            pose,
            epoch=f"e{epoch}_b{idx}",
            out=render_out,
            name=name[idx] if name else None,
            sound=sound,
            contact=contacts[idx] if contacts is not None else None
        )
    
    p_map(render_worker, enumerate(poses))

def _save_fk_data(self, rotations, positions, names, epoch, fk_out, poses):
    """Save forward kinematics data to pickle files."""
    Path(fk_out).mkdir(parents=True, exist_ok=True)
    
    for idx, (rot, pos, name) in enumerate(zip(rotations, positions, names)):
        output_path = Path(fk_out) / f"{epoch}_{idx}_{Path(name).stem}.pkl"
        
        with output_path.open("wb") as f:
            pickle.dump({
                "smpl_poses": rot.reshape(-1, 72).cpu().numpy(),
                "smpl_trans": pos.cpu().numpy(),
                "full_pose": poses[idx]
            }, f)

def _save_long_fk_data(self, rotations, positions, names, epoch, fk_out, poses):
    """Save long sequence FK data to pickle files."""
    Path(fk_out).mkdir(parents=True, exist_ok=True)
    output_path = Path(fk_out) / f"{epoch}_{Path(names[0]).stem}.pkl"
    
    with output_path.open("wb") as f:
        pickle.dump({
            "smpl_poses": rotations.reshape(-1, 72).cpu().numpy(),
            "smpl_trans": positions.reshape(-1, 3).cpu().numpy(),
            "full_pose": poses[0]
        }, f)