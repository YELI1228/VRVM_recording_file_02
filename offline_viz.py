#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import os

import numpy as np
import matplotlib.pyplot as plt

import rosbag

from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from tf2_msgs.msg import TFMessage
from sensor_msgs import point_cloud2  # 用来从 PointCloud2 读取点


# ======================== 配置区（可调参数） ========================

# 要读的 bag 文件名
BAG_FILE = "offline_viz_demo.bag"

# 帧选择模式:
#   "last"  - 取每个 topic 的最后一条消息（默认）
#   "time"  - 取接近目标时间的那一帧（从 bag 开始算起 N 秒）
FRAME_MODE = "last"   # 可改成 "time"

# 仅当 FRAME_MODE == "time" 时生效：
# 从 bag 起始时间算起多少秒的位置截帧，例如 30.0 表示起始 + 30 秒处
TARGET_TIME_SEC_FROM_START = 30.0

# 话题名称（根据你当前 rostopic list）
TOPIC_LIO_MAP      = "/lio_sam/mapping/cloud_registered"
TOPIC_LIO_PATH     = "/lio_sam/mapping/path"
TOPIC_VMAP_MARKERS = "/virtual_map/markers"
TOPIC_RRT_MARKERS  = "/em_planner/markers"
TOPIC_OCC_GRID     = "/projected_map"
TOPIC_TF           = "/tf"
TOPIC_TF_STATIC    = "/tf_static"

# 坐标系（需要和 TF 里的名字对应，如果不对你可自行改）
GLOBAL_FRAME = "map"             # 占据栅格和 LIO 通常在 map 下
BOAT_FRAME   = "wamv/base_link"  # USV 机体 TF 子坐标系（如果不对，改成实际的）

# 是否绘制各个图层（可以快速关掉某些层看效果）
DRAW_OCCUPANCY = True
DRAW_LIO_MAP   = True
DRAW_LIO_PATH  = True
DRAW_VMAP      = True
DRAW_RRT       = True
DRAW_BOAT      = True

# OccupancyGrid 灰度 & colormap
COLOR_OCC_FREE     = 1.0     # 空闲栅格 灰度(白)
COLOR_OCC_OCCUPIED = 0.0     # 占据栅格 灰度(黑)
COLOR_OCC_UNKNOWN  = 0.5     # 未知栅格 灰度(中灰)
OCC_CMAP           = "gray"  # imshow 使用的 colormap，如 "gray", "Greys", "viridis" 等
OCC_ALPHA          = 0.7     # 地图透明度

# 颜色配置（其他层）
COLOR_LIO_MAP       = "#aaaaaa"    # LIO 点云颜色
COLOR_LIO_PATH      = "#1f77b4"    # LIO 路径颜色 (蓝)
COLOR_VMAP_ELLIPSE  = "black"      # 虚拟地标椭圆轮廓颜色
COLOR_VMAP_FIXED    = "#ff7f0e"    # 固定节点（SPHERE）颜色
COLOR_RRT_PATH      = "#8000ff"    # RRT 路径颜色 (紫)
COLOR_BOAT          = "#d62728"    # USV 船体颜色 (红)

# 线宽/点大小
VMAP_ELLIPSE_LINEWIDTH = 0.7
RRT_LINEWIDTH          = 1.5
LIO_PATH_LINEWIDTH     = 1.5
LIO_MAP_POINT_SIZE     = 1.0

# LIO 点云下采样：每 N 个点取一个，避免图太重
LIO_MAP_DOWNSAMPLE = 20

# 船体简化模型（三角形）的尺寸
BOAT_LENGTH = 3.0   # m
BOAT_WIDTH  = 1.5   # m

# 输出图片文件名
OUTPUT_FIG_NAME = "offline_viz_output.png"

# ======================================================================


def yaw_from_quat(q):
    """从 geometry_msgs/Quaternion 计算 yaw（绕 z 轴）"""
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def draw_boat(ax, x, y, yaw,
              length=BOAT_LENGTH, width=BOAT_WIDTH,
              color=COLOR_BOAT):
    """在 ax 上画一个简化的船体三角形 (顶视图)"""
    # 局部坐标系下：船头在 +x，船尾两点在 -x，两侧
    pts_local = np.array([
        [ length / 2.0,  0.0],
        [-length / 2.0, -width / 2.0],
        [-length / 2.0,  width / 2.0],
        [ length / 2.0,  0.0]
    ])

    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    pts_world = (R @ pts_local.T).T + np.array([x, y])

    ax.plot(pts_world[:, 0], pts_world[:, 1],
            color=color, linewidth=2.0, label="USV")


def main():
    if not os.path.exists(BAG_FILE):
        print("找不到 bag 文件:", BAG_FILE)
        return

    print("读取 bag 文件:", BAG_FILE)
    bag = rosbag.Bag(BAG_FILE)

    # 各种数据的“选中帧”
    lio_cloud_msg = None
    lio_path_msg  = None
    vmap_markers_msg = None
    rrt_markers_msg  = None
    occ_grid_msg  = None
    boat_pose = None  # (x, y, yaw)

    # 对于 FRAME_MODE=="time"，先拿到 bag 起始时间
    if FRAME_MODE == "time":
        bag_start = bag.get_start_time()   # float 秒
        target_time = bag_start + TARGET_TIME_SEC_FROM_START
        print(f"[INFO] FRAME_MODE='time', bag_start={bag_start:.3f}, "
              f"target_time={target_time:.3f}")
    else:
        target_time = None

    # 为 "time" 模式记录各 topic 当前选中帧的时间戳（方便比较哪个更近）
    times_lio_cloud = None
    times_lio_path  = None
    times_vmap      = None
    times_rrt       = None
    times_occ       = None
    times_boat      = None

    # 遍历 bag 消息
    for topic, msg, t in bag.read_messages():
        t_sec = t.to_sec()

        if FRAME_MODE == "last":
            # 和最简单版本一样：只保留每个 topic 的最后一条消息
            if topic == TOPIC_LIO_MAP:
                lio_cloud_msg = msg
            elif topic == TOPIC_LIO_PATH:
                lio_path_msg = msg
            elif topic == TOPIC_VMAP_MARKERS:
                vmap_markers_msg = msg
            elif topic == TOPIC_RRT_MARKERS:
                rrt_markers_msg = msg
            elif topic == TOPIC_OCC_GRID:
                occ_grid_msg = msg
            elif topic in (TOPIC_TF, TOPIC_TF_STATIC):
                if isinstance(msg, TFMessage):
                    for tr in msg.transforms:
                        if tr.child_frame_id == BOAT_FRAME and tr.header.frame_id == GLOBAL_FRAME:
                            x = tr.transform.translation.x
                            y = tr.transform.translation.y
                            yaw = yaw_from_quat(tr.transform.rotation)
                            boat_pose = (x, y, yaw)

        elif FRAME_MODE == "time":
            # 只考虑 t_sec <= target_time 的消息，取离 target_time 最近的那一条
            if t_sec > target_time:
                continue

            if topic == TOPIC_LIO_MAP:
                if (times_lio_cloud is None) or (t_sec > times_lio_cloud):
                    times_lio_cloud = t_sec
                    lio_cloud_msg = msg

            elif topic == TOPIC_LIO_PATH:
                if (times_lio_path is None) or (t_sec > times_lio_path):
                    times_lio_path = t_sec
                    lio_path_msg = msg

            elif topic == TOPIC_VMAP_MARKERS:
                if (times_vmap is None) or (t_sec > times_vmap):
                    times_vmap = t_sec
                    vmap_markers_msg = msg

            elif topic == TOPIC_RRT_MARKERS:
                if (times_rrt is None) or (t_sec > times_rrt):
                    times_rrt = t_sec
                    rrt_markers_msg = msg

            elif topic == TOPIC_OCC_GRID:
                if (times_occ is None) or (t_sec > times_occ):
                    times_occ = t_sec
                    occ_grid_msg = msg

            elif topic in (TOPIC_TF, TOPIC_TF_STATIC):
                if isinstance(msg, TFMessage):
                    for tr in msg.transforms:
                        if tr.child_frame_id == BOAT_FRAME and tr.header.frame_id == GLOBAL_FRAME:
                            if (times_boat is None) or (t_sec > times_boat):
                                times_boat = t_sec
                                x = tr.transform.translation.x
                                y = tr.transform.translation.y
                                yaw = yaw_from_quat(tr.transform.rotation)
                                boat_pose = (x, y, yaw)

    bag.close()

    # ======================= 开始画图 =======================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)

    all_x = []
    all_y = []

    # -------- 1) OccupancyGrid 作为背景 --------
    if DRAW_OCCUPANCY and occ_grid_msg is not None:
        info = occ_grid_msg.info
        w = info.width
        h = info.height
        res = info.resolution

        data = np.array(occ_grid_msg.data, dtype=np.int8).reshape((h, w))
        # 构造一张灰度图：白=free，黑=occupied，中灰=unknown
        img = np.zeros((h, w), dtype=np.float32)
        img[data == -1] = COLOR_OCC_UNKNOWN
        img[(data >= 0) & (data < 50)] = COLOR_OCC_FREE
        img[data >= 50] = COLOR_OCC_OCCUPIED

        # origin 是地图左下角，图像用 origin="lower" 才和 RViz 一致
        x_min = info.origin.position.x
        y_min = info.origin.position.y
        x_max = x_min + w * res
        y_max = y_min + h * res

        extent = (x_min, x_max, y_min, y_max)

        ax.imshow(
            img,
            cmap=OCC_CMAP,
            origin="lower",
            extent=extent,
            alpha=OCC_ALPHA
        )

        all_x += [x_min, x_max]
        all_y += [y_min, y_max]

    # -------- 2) LIO-SAM 点云地图 --------
    if DRAW_LIO_MAP and lio_cloud_msg is not None:
        xs_pc = []
        ys_pc = []
        for i, p in enumerate(
            point_cloud2.read_points(
                lio_cloud_msg,
                field_names=("x", "y"),
                skip_nans=True
            )
        ):
            if i % LIO_MAP_DOWNSAMPLE != 0:
                continue
            xs_pc.append(p[0])
            ys_pc.append(p[1])

        ax.scatter(
            xs_pc,
            ys_pc,
            s=LIO_MAP_POINT_SIZE,
            c=COLOR_LIO_MAP,
            alpha=0.5,
            label="LIO-SAM map"
        )

        all_x += xs_pc
        all_y += ys_pc

    # -------- 3) LIO-SAM 轨迹 Path --------
    if DRAW_LIO_PATH and lio_path_msg is not None:
        xs_path = [p.pose.position.x for p in lio_path_msg.poses]
        ys_path = [p.pose.position.y for p in lio_path_msg.poses]

        ax.plot(
            xs_path,
            ys_path,
            color=COLOR_LIO_PATH,
            linewidth=LIO_PATH_LINEWIDTH,
            label="LIO-SAM path"
        )

        all_x += xs_path
        all_y += ys_path

    # -------- 4) virtual_map 椭圆 + 固定节点 --------
    if DRAW_VMAP and vmap_markers_msg is not None:
        for mk in vmap_markers_msg.markers:
            # 椭圆轮廓：你 C++ 里用 LINE_STRIP 画的协方差椭圆
            if mk.type == Marker.LINE_STRIP:
                xs = [pt.x for pt in mk.points]
                ys = [pt.y for pt in mk.points]
                if xs and ys:
                    ax.plot(
                        xs,
                        ys,
                        color=COLOR_VMAP_ELLIPSE,
                        linewidth=VMAP_ELLIPSE_LINEWIDTH,
                        alpha=0.9
                    )
                    all_x += xs
                    all_y += ys

            # 固定节点（占据的 actual landmark）：用 SPHERE 画的小圆
            elif mk.type == Marker.SPHERE:
                cx = mk.pose.position.x
                cy = mk.pose.position.y
                ax.scatter(
                    [cx],
                    [cy],
                    s=30,
                    edgecolors=COLOR_VMAP_FIXED,
                    facecolors="none",
                    linewidths=1.2
                )
                all_x.append(cx)
                all_y.append(cy)

    # -------- 5) RRT 路径 (MarkerArray) --------
    if DRAW_RRT and rrt_markers_msg is not None:
        # em_planner/markers 里一般会有好几个 Marker，
        # 我们把所有 LINE_STRIP 都画出来（RRT 树/生成路径）
        for mk in rrt_markers_msg.markers:
            if mk.type == Marker.LINE_STRIP and mk.points:
                xs = [pt.x for pt in mk.points]
                ys = [pt.y for pt in mk.points]
                ax.plot(
                    xs,
                    ys,
                    color=COLOR_RRT_PATH,
                    linewidth=RRT_LINEWIDTH,
                    alpha=0.9,
                    label="RRT path"
                )
                all_x += xs
                all_y += ys

    # -------- 6) USV 船体位置 (从 TF 里取 map->wamv/base_link) --------
    if DRAW_BOAT and boat_pose is not None:
        bx, by, byaw = boat_pose
        draw_boat(ax, bx, by, byaw)
        all_x.append(bx)
        all_y.append(by)

    # -------- 7) 视野范围 & 图例 --------
    if all_x and all_y:
        margin = 5.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_NAME, dpi=300)
    print("已保存图片:", OUTPUT_FIG_NAME)
    plt.show()


if __name__ == "__main__":
    main()

