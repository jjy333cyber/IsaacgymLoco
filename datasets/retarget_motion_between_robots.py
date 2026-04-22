#!/usr/bin/env python3
"""基于“重定向输出 motion 格式”再重定向到另一个机器人。

你的 `datasets/retarget_kp_motions.py` 会输出一个 JSON（扩展名 .txt），其中 `Frames` 每帧 61 维：
[root_pos(3), root_rot(4), joint_pose(12), toe_local(12), lin_vel(3), ang_vel(3), joint_vel(12), toe_vel(12)]。

这个脚本把该输出当作“中间表示”，再生成另一台机器人的同格式 motion：

- 读取源 motion（src_motion_file）
- 用源机器人 URDF + (root_pos/root_rot/joint_pose) 做 FK，得到 4 个脚端的世界坐标 toe_pos_world
- 计算参考根姿态（把 INIT_ROT / SIM_ROOT_OFFSET 去掉），再映射到目标机器人坐标（乘回 dst 的 INIT_ROT/offset）
- 在目标机器人上做 IK：给定 toe_pos_world（世界系目标），解出目标 joint_pose
- 由 base pose + joint_pose 计算 toe_local（在 base frame 下）
- 重新用差分计算速度项，写回 JSON motion

限制/假设：
- 源 motion 的 joint_pose(12) 与源 URDF 的“可动关节顺序”一致。
- 配置模块需提供：URDF_FILENAME, INIT_ROT, SIM_ROOT_OFFSET, SIM_TOE_JOINT_IDS, JOINT_DAMPING, DEFAULT_JOINT_POSE。
- 脚端目标来自源机器人的脚端世界轨迹；若目标机器人腿长/几何差异过大，IK 可能失败或产生不合理姿态。

用法示例：
  python datasets/retarget_motion_between_robots.py \
    --src_motion datasets/mocap_motions_a1/trot0.txt \
    --src_config datasets.retarget_config_lite3 \
    --dst_config datasets.retarget_config_aliengo \
    --out_motion /tmp/trot0_aliengo.txt

也支持把 --src_motion 指定为目录：
    python datasets/retarget_motion_between_robots.py \
        --src_motion datasets/mocap_motions_a1 \
        --src_config datasets.retarget_config_a1 \
        --dst_config datasets.retarget_config_aliengo

当 --out_motion 未指定时，默认输出到目标 config 的 OUTPUT_DIR（位于工作区 datasets 下）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
    import pybullet
    import pybullet_data
    from pybullet_utils import transformations
except ModuleNotFoundError as e:
    missing = getattr(e, "name", str(e))
    raise ModuleNotFoundError(
        f"缺少依赖模块：{missing}。\n"
        "请确认你在包含依赖的 Python 环境中运行（例如先 `conda activate himamp`），"
        "或安装 requirements 中的依赖（至少需要 numpy + pybullet）。"
    ) from e


def _load_motion_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _save_motion_json(path: Path, *, frames: np.ndarray, frame_duration: float, motion_weight: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "LoopMode": "Wrap",
        "FrameDuration": float(frame_duration),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": float(motion_weight),
        "Frames": frames.tolist(),
    }
    with path.open("w") as f:
        json.dump(out, f)


def _import_config_module(module_name: str):
    """兼容两种写法：datasets.retarget_config_xxx 或直接 retarget_config_xxx。"""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if module_name.startswith("datasets."):
            raise
        return importlib.import_module(f"datasets.{module_name}")


def _iter_motion_files(src_motion: Path) -> List[Path]:
    """src_motion 可以是单文件或目录；目录下默认取所有 .txt 文件。"""
    if src_motion.is_file():
        return [src_motion]
    if not src_motion.is_dir():
        raise FileNotFoundError(f"src_motion 不存在：{src_motion}")

    files = sorted([p for p in src_motion.iterdir() if p.is_file() and p.suffix == ".txt"])
    return files


def _default_out_dir(dst_cfg) -> Path:
    """默认输出目录：优先使用目标 config 的 OUTPUT_DIR（通常在 datasets/ 下）。"""
    out_dir = getattr(dst_cfg, "OUTPUT_DIR", None)
    if out_dir is None:
        # fallback：本脚本同级目录下创建子目录（也在 datasets/ 下）
        return Path(__file__).resolve().parent / "mocap_motions_converted"
    return Path(str(out_dir))


def _is_motion_json(obj: dict) -> bool:
    return isinstance(obj, dict) and "Frames" in obj and isinstance(obj["Frames"], list)


def _quat_norm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return transformations.quaternion_multiply(q1, q2)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return transformations.quaternion_inverse(q)


def _get_movable_joint_ids(robot_id: int) -> List[int]:
    ids: List[int] = []
    num = pybullet.getNumJoints(robot_id)
    for j in range(num):
        joint_type = pybullet.getJointInfo(robot_id, j)[2]
        if joint_type in (pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC):
            ids.append(j)
    return ids


def _decode_name(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def _get_movable_joint_names(robot_id: int, joint_ids: List[int]) -> List[str]:
    names: List[str] = []
    for j in joint_ids:
        info = pybullet.getJointInfo(robot_id, j)
        names.append(_decode_name(info[1]))
    return names


def _find_link_id_by_name(robot_id: int, link_name: str) -> int | None:
    """在 PyBullet 中，link index 与其父 joint index 对应；返回该 link 的 index。"""
    for j in range(pybullet.getNumJoints(robot_id)):
        info = pybullet.getJointInfo(robot_id, j)
        this_link_name = _decode_name(info[12])
        if this_link_name == link_name:
            return j
    return None


def _extract_leg_prefix(name: str) -> str | None:
    """从 joint/link 名称里提取腿前缀：FL/FR/RL/RR/HL/HR。"""
    name = str(name)
    for p in ("FL", "FR", "RL", "RR", "HL", "HR"):
        if name.startswith(p + "_") or name.startswith(p):
            return p
    return None


def _leg_slot(prefix: str) -> str | None:
    """把不同 URDF 的后腿命名统一到槽位：FL/FR/HL/HR。"""
    if prefix in ("FL", "FR", "HL", "HR"):
        return prefix
    if prefix == "RL":
        return "HL"
    if prefix == "RR":
        return "HR"
    return None


def _build_slot_map_from_ids(robot_id: int, ids: List[int], values: List[np.ndarray] | None = None) -> dict[str, int] | dict[str, np.ndarray]:
    """根据 joint id 的名字，把列表重排到槽位顺序（FL/FR/HL/HR）。

    - ids 通常来自 config 的 SIM_*_JOINT_IDS（其顺序可能是 FR/FL/RR/RL 等）
    - values 若提供，则认为与 ids 同步（比如 SIM_TOE_OFFSET_LOCAL）
    """
    out: dict[str, int] | dict[str, np.ndarray]
    out = {}
    for i, jid in enumerate(ids):
        info = pybullet.getJointInfo(robot_id, int(jid))
        joint_name = _decode_name(info[1])
        link_name = _decode_name(info[12])
        prefix = _extract_leg_prefix(joint_name) or _extract_leg_prefix(link_name)
        if prefix is None:
            continue
        slot = _leg_slot(prefix)
        if slot is None:
            continue
        if values is None:
            out[slot] = int(jid)
        else:
            out[slot] = np.asarray(values[i], dtype=np.float64)
    return out


def _resolve_hip_links_by_slot(robot_id: int, cfg) -> List[int]:
    """返回四个髋 link id，按槽位顺序 [FL, FR, HL, HR]。"""
    hip_ids = list(getattr(cfg, "SIM_HIP_JOINT_IDS"))
    slot_map = _build_slot_map_from_ids(robot_id, hip_ids, values=None)
    if all(k in slot_map for k in ("FL", "FR", "HL", "HR")):
        return [int(slot_map["FL"]), int(slot_map["FR"]), int(slot_map["HL"]), int(slot_map["HR"])]
    # fallback：假设 config 已经是 [FL, FR, HL/RL, HR/RR]
    return hip_ids


def _resolve_toe_offset_local_by_slot(robot_id: int, cfg) -> List[np.ndarray]:
    """返回四个脚端偏移（local），按槽位顺序 [FL, FR, HL, HR]。"""
    toe_ids = list(getattr(cfg, "SIM_TOE_JOINT_IDS"))
    toe_off = list(getattr(cfg, "SIM_TOE_OFFSET_LOCAL"))
    slot_map = _build_slot_map_from_ids(robot_id, toe_ids, values=toe_off)
    if all(k in slot_map for k in ("FL", "FR", "HL", "HR")):
        return [slot_map["FL"], slot_map["FR"], slot_map["HL"], slot_map["HR"]]
    # fallback：假设 config 已经是 [FL, FR, HL/RL, HR/RR]
    return [np.asarray(v, dtype=np.float64) for v in toe_off]


def _quat_rotate_vec(q_xyzw: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """用四元数旋转向量（不含平移）。"""
    p, _ = pybullet.multiplyTransforms([0, 0, 0], q_xyzw.tolist(), v_xyz.tolist(), [0, 0, 0, 1])
    return np.asarray(p, dtype=np.float64)


def _calc_heading_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """仿照 retarget_utils.calc_heading_rot：提取绕 z 轴的 yaw，并返回对应四元数。"""
    # 通过把 forward=[1,0,0] 旋转到世界系来求 heading
    fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rot_fwd = _quat_rotate_vec(q_xyzw, fwd)
    heading = float(np.arctan2(rot_fwd[1], rot_fwd[0]))
    return transformations.quaternion_about_axis(heading, [0, 0, 1])


def _resolve_toe_links_canonical(robot_id: int, cfg) -> List[int]:
    """按 canonical 顺序返回四个脚端 link id。

    canonical 顺序约定：FL, FR, (RL or HL), (RR or HR)
    - Unitree/A1/Go2 通常是 RL/RR
    - Lite3 URDF 常用 HL/HR 表示后腿
    """
    fl = getattr(cfg, "FL_FOOT_NAME", None)
    fr = getattr(cfg, "FR_FOOT_NAME", None)
    hl = getattr(cfg, "HL_FOOT_NAME", None)
    hr = getattr(cfg, "HR_FOOT_NAME", None)
    if fl is None or fr is None or hl is None or hr is None:
        return list(getattr(cfg, "SIM_TOE_JOINT_IDS"))

    names = [str(fl), str(fr), str(hl), str(hr)]
    ids: List[int] = []
    for name in names:
        link_id = _find_link_id_by_name(robot_id, name)
        if link_id is None:
            # fallback：保持原实现（依赖 config 中的 link id 顺序）
            return list(getattr(cfg, "SIM_TOE_JOINT_IDS"))
        ids.append(int(link_id))
    return ids


def _build_canonical_joint_names(movable_joint_names: List[str]) -> List[str] | None:
    """根据 URDF 关节命名猜测 canonical DOF 名称序列（长度=12）。

    canonical DOF 顺序与 legged_gym/IsaacGym 常用一致：
    - Unitree 命名：FL/FR/RL/RR + hip/thigh/calf
    - Lite3 命名：FL/FR/HL/HR + HipX/HipY/Knee
    """
    names_set = set(movable_joint_names)

    # Unitree 风格（A1/Go1/Go2/Aliengo 等）
    if any(n.endswith("_hip_joint") for n in movable_joint_names):
        hind_left = "RL" if any(n.startswith("RL_") for n in movable_joint_names) else "HL"
        hind_right = "RR" if any(n.startswith("RR_") for n in movable_joint_names) else "HR"
        legs = ["FL", "FR", hind_left, hind_right]
        parts = ["hip_joint", "thigh_joint", "calf_joint"]
        canonical = [f"{leg}_{part}" for leg in legs for part in parts]
        if set(canonical).issubset(names_set):
            return canonical

    # Lite3 风格
    if any("HipX_joint" in n for n in movable_joint_names):
        hind_left = "HL" if any(n.startswith("HL_") for n in movable_joint_names) else "RL"
        hind_right = "HR" if any(n.startswith("HR_") for n in movable_joint_names) else "RR"
        legs = ["FL", "FR", hind_left, hind_right]
        parts = ["HipX_joint", "HipY_joint", "Knee_joint"]
        canonical = [f"{leg}_{part}" for leg in legs for part in parts]
        if set(canonical).issubset(names_set):
            return canonical

    return None


def _build_joint_permutations(robot_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """建立 canonical(IsaacGym) <-> PyBullet(movable joint order) 的置换。

    返回：
      bullet_from_canon: shape(12,), bullet_pose = canon_pose[bullet_from_canon]
      canon_from_bullet: shape(12,), canon_pose = bullet_pose[canon_from_bullet]
    """
    movable = _get_movable_joint_ids(robot_id)
    if len(movable) != 12:
        return None
    bullet_names = _get_movable_joint_names(robot_id, movable)
    canonical_names = _build_canonical_joint_names(bullet_names)
    if canonical_names is None or len(canonical_names) != 12:
        return None

    name_to_canon = {name: i for i, name in enumerate(canonical_names)}
    try:
        bullet_from_canon = np.asarray([name_to_canon[n] for n in bullet_names], dtype=np.int64)
    except KeyError:
        return None

    canon_from_bullet = np.empty((12,), dtype=np.int64)
    for bullet_i, canon_i in enumerate(bullet_from_canon.tolist()):
        canon_from_bullet[int(canon_i)] = int(bullet_i)
    return bullet_from_canon, canon_from_bullet


def _set_robot_base_and_joints(robot_id: int, base_pos: np.ndarray, base_rot: np.ndarray, joint_pose: np.ndarray) -> None:
    pybullet.resetBasePositionAndOrientation(robot_id, base_pos.tolist(), base_rot.tolist())

    movable = _get_movable_joint_ids(robot_id)
    if len(movable) != int(joint_pose.shape[0]):
        raise ValueError(
            f"URDF movable joints={len(movable)} but joint_pose has {joint_pose.shape[0]} values. "
            "请确认 motion 的 joint_pose 维度与 URDF 可动关节数一致。"
        )

    for idx, joint_id in enumerate(movable):
        pybullet.resetJointState(robot_id, joint_id, float(joint_pose[idx]))


def _link_world_pos(robot_id: int, link_id: int) -> np.ndarray:
    state = pybullet.getLinkState(robot_id, link_id, computeForwardKinematics=True)
    return np.asarray(state[4], dtype=np.float64)


def _toe_local_in_base(base_pos: np.ndarray, base_rot: np.ndarray, toe_world: np.ndarray) -> np.ndarray:
    inv_pos, inv_rot = pybullet.invertTransform(base_pos.tolist(), base_rot.tolist())
    toe_local, _ = pybullet.multiplyTransforms(inv_pos, inv_rot, toe_world.tolist(), [0, 0, 0, 1])
    return np.asarray(toe_local, dtype=np.float64)


def _split_src_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_pos = frame[0:3]
    root_rot = frame[3:7]
    joint_pose = frame[7:19]
    return root_pos, root_rot, joint_pose


def _compute_linear_vel_base(curr_pos: np.ndarray, curr_rot: np.ndarray, next_pos: np.ndarray, dt: float) -> np.ndarray:
    v_world = (next_pos - curr_pos) / dt
    r = np.array(pybullet.getMatrixFromQuaternion(curr_rot.tolist()), dtype=np.float64).reshape(3, 3)
    # 与 retarget_kp_motions.py 一致：row-vector 右乘 => 等价于 R^T * v_world
    return v_world @ r


def _compute_angular_vel_base(curr_rot: np.ndarray, next_rot: np.ndarray, init_rot_xyzw: np.ndarray, dt: float) -> np.ndarray:
    delta_q = pybullet.getDifferenceQuaternion(curr_rot.tolist(), next_rot.tolist())
    axis, angle = pybullet.getAxisAngleFromQuaternion(delta_q)
    w_world = np.asarray(axis, dtype=np.float64) * (float(angle) / dt)

    inv_init = _quat_inv(init_rot_xyzw)
    _, base_from_init = pybullet.multiplyTransforms([0, 0, 0], inv_init.tolist(), [0, 0, 0], curr_rot.tolist())
    _, inv_base = pybullet.invertTransform([0, 0, 0], base_from_init)
    w_base, _ = pybullet.multiplyTransforms([0, 0, 0], inv_base, w_world.tolist(), [0, 0, 0, 1])
    return np.asarray(w_base, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_motion",
        type=str,
        required=True,
        help="源 motion 文件或目录（output_motion 生成的 JSON .txt；目录下会批量转换所有 .txt）",
    )
    parser.add_argument(
        "--src_config",
        type=str,
        required=True,
        help="源机器人 config 模块，例如 datasets.retarget_config_lite3",
    )
    parser.add_argument(
        "--dst_config",
        type=str,
        required=True,
        help="目标机器人 config 模块，例如 datasets.retarget_config_aliengo",
    )
    parser.add_argument(
        "--out_motion",
        type=str,
        default=None,
        help="输出目标 motion 文件路径；若不填，默认输出到目标 config 的 OUTPUT_DIR 并沿用原文件名",
    )
    parser.add_argument("--motion_weight", type=float, default=None, help="覆盖输出 MotionWeight；默认沿用源 motion")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="用 PyBullet GUI 可视化转换过程（会按帧刷新，速度约等于 FrameDuration）",
    )
    args = parser.parse_args()

    src_motion = Path(args.src_motion)
    src_cfg = _import_config_module(args.src_config)
    dst_cfg = _import_config_module(args.dst_config)

    src_files = _iter_motion_files(src_motion)
    if len(src_files) == 0:
        raise FileNotFoundError(f"目录下没有 .txt motion 文件：{src_motion}")

    out_dir = _default_out_dir(dst_cfg)

    # 连接 pybullet
    connection_mode = pybullet.GUI if args.visualize else pybullet.DIRECT
    pybullet.connect(connection_mode)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, -9.8)
    pybullet.loadURDF("plane_implicit.urdf")

    src_robot = pybullet.loadURDF(
        str(src_cfg.URDF_FILENAME),
        [0, 0, 0.3],
        [0, 0, 0, 1],
        flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
    )
    dst_robot = pybullet.loadURDF(
        str(dst_cfg.URDF_FILENAME),
        [0, 0, 0.3],
        [0, 0, 0, 1],
        flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
    )

    src_init = _quat_norm(np.asarray(src_cfg.INIT_ROT, dtype=np.float64))
    dst_init = _quat_norm(np.asarray(dst_cfg.INIT_ROT, dtype=np.float64))
    inv_src_init = _quat_inv(src_init)

    src_root_offset = np.asarray(src_cfg.SIM_ROOT_OFFSET, dtype=np.float64)
    dst_root_offset = np.asarray(dst_cfg.SIM_ROOT_OFFSET, dtype=np.float64)

    # 参考 retarget_kp_motions.py：脚端目标用“髋位置 + (hip->toe delta) + toe_offset_world”，
    # 因此需要稳定地拿到每条腿的髋 link id，并把 config 中可能的 FR/FL/RR/RL 顺序重排到槽位顺序。
    src_hip_links = _resolve_hip_links_by_slot(src_robot, src_cfg)
    dst_hip_links = _resolve_hip_links_by_slot(dst_robot, dst_cfg)
    dst_toe_offset_local_by_slot = _resolve_toe_offset_local_by_slot(dst_robot, dst_cfg)

    # 统一用 canonical(FL,FR,HL/RL,HR/RR) 脚端顺序解析 toe link。
    # 这样不同 config 的 SIM_TOE_JOINT_IDS 顺序不一致时也不会左右/前后对调。
    src_toe_links = _resolve_toe_links_canonical(src_robot, src_cfg)
    dst_toe_links = _resolve_toe_links_canonical(dst_robot, dst_cfg)

    # 关节顺序：motion 侧采用 canonical（与 IsaacGym/legged_gym 的 dof_names 常用顺序一致），
    # PyBullet IK/FK 侧采用它自己的 movable joint 顺序。两者用关节名做置换。
    src_joint_perm = _build_joint_permutations(src_robot)
    dst_joint_perm = _build_joint_permutations(dst_robot)
    if src_joint_perm is None or dst_joint_perm is None:
        print(
            "[warn] 未能根据关节名建立 canonical↔PyBullet 的关节置换，将退回到原始顺序。"
        )
    src_bullet_from_canon = src_joint_perm[0] if src_joint_perm is not None else None
    dst_bullet_from_canon = dst_joint_perm[0] if dst_joint_perm is not None else None
    dst_canon_from_bullet = dst_joint_perm[1] if dst_joint_perm is not None else None

    # 可视化时把两台机器人分开放置，避免重叠。
    # 注意：该偏移只用于显示；速度/输出 pose 都仍使用“未偏移”的世界坐标。
    if args.visualize:
        src_vis_offset = np.array([0.0, -0.8, 0.0], dtype=np.float64)
        dst_vis_offset = np.array([0.0, +0.8, 0.0], dtype=np.float64)
        toe_vis_delta = dst_vis_offset - src_vis_offset  # src 区域的 toe 目标平移到 dst 区域
        # 让相机看向两台机器人中间
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[0.0, 0.0, 0.3],
        )
    else:
        src_vis_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        dst_vis_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        toe_vis_delta = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    converted = 0
    skipped = 0

    for src_file in src_files:
        try:
            src_data = _load_motion_json(src_file)
        except Exception:
            skipped += 1
            continue

        if not _is_motion_json(src_data):
            skipped += 1
            continue

        src_frames = np.asarray(src_data["Frames"], dtype=np.float64)
        if src_frames.ndim != 2 or src_frames.shape[1] < 19:
            skipped += 1
            continue

        dt = float(src_data.get("FrameDuration", 0.02))
        motion_weight = (
            float(src_data.get("MotionWeight", 1.0))
            if args.motion_weight is None
            else float(args.motion_weight)
        )

        # 目标输出文件名
        if args.out_motion is not None:
            out_motion_path = Path(args.out_motion)
        else:
            out_motion_path = out_dir / src_file.name

        # 先把每帧的 dst pose（31 维，不含速度）算出来
        M = int(src_frames.shape[0])
        dst_pose31 = np.zeros((M, 31), dtype=np.float64)

        for i in range(M):
            src_root_pos, src_root_rot, src_joint = _split_src_frame(src_frames[i])
            src_root_rot = _quat_norm(src_root_rot)

            # motion 输入的 joint_pose 视为 canonical 顺序；喂给 PyBullet 之前重排到 bullet 顺序
            src_joint_canon = np.asarray(src_joint, dtype=np.float64)
            if src_bullet_from_canon is not None:
                src_joint_bullet = src_joint_canon[src_bullet_from_canon]
            else:
                src_joint_bullet = src_joint_canon

            # 参考根（去掉 src 的 offset / init）
            ref_root_pos = src_root_pos - src_root_offset
            ref_root_rot = _quat_mul(src_root_rot, inv_src_init)
            ref_root_rot = _quat_norm(ref_root_rot)

            # 映射到 dst（乘回 dst 的 init / offset）
            dst_root_pos = ref_root_pos + dst_root_offset
            dst_root_rot = _quat_mul(ref_root_rot, dst_init)
            dst_root_rot = _quat_norm(dst_root_rot)

            # 用源状态做 FK，提取 toe world。
            # - 非可视化：直接在原点坐标系下算
            # - 可视化：把源机器人放到左侧显示区域算 FK，再把 toe world 整体平移到右侧
            src_base_for_fk = src_root_pos + (src_vis_offset if args.visualize else 0.0)
            _set_robot_base_and_joints(src_robot, src_base_for_fk, src_root_rot, src_joint_bullet)
            toe_world_src = [_link_world_pos(src_robot, link_id) for link_id in src_toe_links]
            hip_world_src = [_link_world_pos(src_robot, link_id) for link_id in src_hip_links]

            # 参考 retarget_kp_motions.py：用 hip->toe 的相对向量作为目标（对平移不敏感）
            hip_toe_delta = [toe - hip for toe, hip in zip(toe_world_src, hip_world_src)]

            # 在目标机器人上 IK
            pybullet.resetBasePositionAndOrientation(
                dst_robot,
                (dst_root_pos + dst_vis_offset).tolist(),
                dst_root_rot.tolist(),
            )

            # retarget_kp_motions.py 的做法：先从 root_rot 得到 heading_rot，再把 toe_offset_local 旋到世界系
            inv_dst_init = _quat_inv(dst_init)
            heading_in = _quat_mul(dst_root_rot, inv_dst_init)
            heading_rot = _calc_heading_rot(heading_in)

            # 当前髋部世界坐标（注意：关节状态沿用上一帧的结果，等价于 keypoint 版本的“隐式连续性”）
            hip_world_dst = [_link_world_pos(dst_robot, link_id) for link_id in dst_hip_links]

            tar_toe_pos: List[np.ndarray] = []
            for leg_idx in range(4):
                toe_offset_world = _quat_rotate_vec(heading_rot, dst_toe_offset_local_by_slot[leg_idx])
                toe_tar = hip_world_dst[leg_idx] + hip_toe_delta[leg_idx]
                # 与 keypoint 版本一致：强制使用参考 toe 的高度（这里参考来自源机器人 FK toe.z）
                toe_tar[2] = toe_world_src[leg_idx][2]
                toe_tar = toe_tar + toe_offset_world
                tar_toe_pos.append(toe_tar)

            # IK 的 damping/restPose 也必须是 PyBullet movable joint 顺序
            dst_damping = np.asarray(dst_cfg.JOINT_DAMPING, dtype=np.float64)
            dst_rest = np.asarray(dst_cfg.DEFAULT_JOINT_POSE, dtype=np.float64)
            if dst_bullet_from_canon is not None and dst_damping.shape[0] == 12:
                dst_damping = dst_damping[dst_bullet_from_canon]
            if dst_bullet_from_canon is not None and dst_rest.shape[0] == 12:
                dst_rest = dst_rest[dst_bullet_from_canon]

            joint_dst = pybullet.calculateInverseKinematics2(
                dst_robot,
                dst_toe_links,
                tar_toe_pos,
                jointDamping=dst_damping.tolist(),
                restPoses=dst_rest.tolist(),
            )
            joint_dst = np.asarray(joint_dst, dtype=np.float64)

            # 输出到 motion 侧统一使用 canonical 顺序
            if dst_canon_from_bullet is not None and joint_dst.shape[0] >= 12:
                joint_dst_canon = joint_dst[:12][dst_canon_from_bullet]
            else:
                joint_dst_canon = joint_dst[:12]

            # 用 IK 结果更新关节，再计算 toe_local（base frame）
            _set_robot_base_and_joints(
                dst_robot,
                dst_root_pos + dst_vis_offset,
                dst_root_rot,
                joint_dst,
            )
            toe_local = [
                _toe_local_in_base(
                    dst_root_pos + dst_vis_offset,
                    dst_root_rot,
                    _link_world_pos(dst_robot, link_id),
                )
                for link_id in dst_toe_links
            ]
            toe_local_flat = np.concatenate(toe_local, axis=0)  # (12,)

            dst_pose31[i] = np.concatenate([dst_root_pos, dst_root_rot, joint_dst_canon, toe_local_flat])

            if args.visualize:
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1
                )
                time.sleep(max(0.0, dt))

        # 计算速度，拼成 61 维
        out_frames = np.zeros((M, 61), dtype=np.float64)
        out_frames[:, 0:31] = dst_pose31

        for i in range(M - 1):
            curr = dst_pose31[i]
            nxt = dst_pose31[i + 1]

            curr_pos = curr[0:3]
            curr_rot = curr[3:7]
            nxt_pos = nxt[0:3]
            nxt_rot = nxt[3:7]

            lin = _compute_linear_vel_base(curr_pos, curr_rot, nxt_pos, dt)
            ang = _compute_angular_vel_base(curr_rot, nxt_rot, dst_init, dt)

            joint_vel = (nxt[7:19] - curr[7:19]) / dt
            toe_vel = (nxt[19:31] - curr[19:31]) / dt

            out_frames[i, 31:34] = lin
            out_frames[i, 34:37] = ang
            out_frames[i, 37:49] = joint_vel
            out_frames[i, 49:61] = toe_vel

        # 最后一帧速度置 0（避免末帧差分不可用）
        out_frames[M - 1, 31:61] = 0.0

        # 与原始输出一致：xy 平移归零
        out_frames[:, 0:2] -= out_frames[0, 0:2]

        _save_motion_json(
            out_motion_path,
            frames=out_frames,
            frame_duration=dt,
            motion_weight=motion_weight,
        )
        converted += 1
        print(
            f"Wrote retargeted motion: {out_motion_path}  "
            f"(frames={out_frames.shape[0]}, dim={out_frames.shape[1]})"
        )

        # 批量模式下如果用户同时给了 out_motion，只处理第一个文件
        if args.out_motion is not None and len(src_files) > 1:
            break

    pybullet.disconnect()

    if skipped > 0:
        print(f"Skipped {skipped} file(s) (non-motion or parse failed).")
    if converted == 0:
        raise RuntimeError("没有成功转换任何 motion 文件")


if __name__ == "__main__":
    main()
