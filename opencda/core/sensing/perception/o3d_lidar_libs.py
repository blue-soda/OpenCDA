# -*- coding: utf-8 -*-
"""
Utility functions for 3d lidar visualization
and processing by utilizing open3d.
"""

# Author: CARLA Team, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import open3d as o3d
import numpy as np

from matplotlib import cm
from scipy.stats import mode

from PIL import Image

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.sensing.perception.obstacle_vehicle import \
    is_vehicle_cococlass, ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import StaticObstacle

# Try to import opencood visualization utils, provide fallbacks if not available
try:
    from opencood.visualization.vis_utils import bbx2oabb, bbx2aabb, bbx2lineset_expand
except ImportError:
    # Provide dummy fallbacks when opencood is not available
    def bbx2oabb(*args, **kwargs):
        return []
    def bbx2aabb(*args, **kwargs):
        return []
    def bbx2lineset_expand(*args, **kwargs):
        return []

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # normalize each channel [0-1] since is what Open3D uses


def o3d_pointcloud_encode(raw_data, point_cloud):
    """
    Encode the raw point cloud(np.array) to Open3d PointCloud object.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw lidar points, (N, 4).

    point_cloud : o3d.PointCloud
        Open3d PointCloud.

    """

    # Isolate the intensity and compute a color for it
    intensity = raw_data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    # int_color = np.c_[
    #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
    #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
    #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
    N = raw_data.shape[0]
    int_color = np.tile([1.0, 1.0, 0.0], (N, 1))  # 黄色

    # Isolate the 3D data
    points = np.array(raw_data[:, :-1], copy=True)
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)


def o3d_visualizer_init(actor_id):
    """
    Initialize the visualizer.

    Parameters
    ----------
    actor_id : int
        Ego vehicle's id.

    Returns
    -------
    vis : o3d.visualizer
        Initialize open3d visualizer.

    """
    vis = o3d.visualization.Visualizer()
    scale_factor = 3
    vis.create_window(window_name=str(actor_id),
                      width=480*scale_factor,
                      height=320*scale_factor,
                      left=480*scale_factor,
                      top=270*scale_factor)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

    return vis


def o3d_visualizer_show(vis, count, point_cloud, objects):
    """
    Visualize the point cloud at runtime.

    Parameters
    ----------
    vis : o3d.Visualizer
        Visualization interface.

    count : int
        Current step since simulation started.

    point_cloud : o3d.PointCloud
        Open3d point cloud.

    objects : dict
        The dictionary containing objects.

    Returns
    -------

    """

    if count == 2:
        vis.add_geometry(point_cloud)

    vis.update_geometry(point_cloud)

    for key, object_list in objects.items():
        # we only draw vehicles for now
        if key != 'vehicles':
            continue
        for object_ in object_list:
            aabb = object_.o3d_bbx
            vis.add_geometry(aabb)

    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    time.sleep(0.001)

    for key, object_list in objects.items():
        if key != 'vehicles':
            continue
        for object_ in object_list:
            aabb = object_.o3d_bbx
            vis.remove_geometry(aabb)

def make_background_transparent(image_path, output_path, bg_color=(255, 0, 255)):
    """
    将指定背景色替换为透明。
    bg_color: RGB tuple in [0, 255]
    """
    if not os.path.exists(image_path):
        print(f"[WARNING] Screenshot file not found: {image_path}")
        return
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    
    # 创建透明 mask：匹配背景色的像素设为透明
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == bg_color[0]) & (green == bg_color[1]) & (blue == bg_color[2])
    
    data[mask] = [0, 0, 0, 0]  # RGBA = transparent
    
    img_transparent = Image.fromarray(data, 'RGBA')
    abs_output_path = os.path.join(BASE_DIR, output_path)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    if os.path.exists(abs_output_path):
        os.remove(abs_output_path)
        print(f"[INFO] Removed existing file: {abs_output_path}")
    else:
        print(f"[INFO] File not found: {abs_output_path}")
    img_transparent.save(abs_output_path)
    del img_transparent 

import os
_spectator_camera = None
BASE_DIR = os.getcwd()
def init_spectator_camera(world, image_size, fov):
    import carla
    global _spectator_camera
    if _spectator_camera is not None:
        return _spectator_camera

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(image_size[0]))
    bp.set_attribute('image_size_y', str(image_size[1]))
    bp.set_attribute('fov', str(fov))
    
    # 初始位置随便设（会被 update 覆盖）
    transform = carla.Transform(carla.Location(z=100))  # 远离场景
    _spectator_camera = world.spawn_actor(bp, transform)
    return _spectator_camera

def capture_spectator_view(world, filename="visualization_output/visualize_spectator_view.png", image_size=(3840, 2160), fov=90):
    """
    Capture an image from the current spectator's viewpoint in CARLA.

    Parameters:
    - world: carla.World instance
    - filename: output image path (PNG)
    - image_size: (width, height) in pixels
    - fov: field of view in degrees

    Returns:
    - True if successful, False otherwise
    """
    # Get current spectator transform
    camera = init_spectator_camera(world, image_size=image_size, fov=fov)
    spectator = world.get_spectator()
    spec_transform = spectator.get_transform()
    camera.set_transform(spec_transform)

    # Variable to store image
    image_data = None

    def save_image(image):
        nonlocal image_data
        image_data = image

    # Listen for one frame
    camera.listen(save_image)

    # Tick the world to trigger sensor update
    world.tick()
    time.sleep(0.1)  # Ensure image is captured

    success = False
    if image_data is not None:
        # Convert CARLA image to numpy array (BGRA -> RGB)
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image_data.height, image_data.width, 4))  # BGRA
        rgb_array = array[:, :, [2, 1, 0]]  # BGR to RGB (ignore alpha)

        # Save as PNG
        img = Image.fromarray(rgb_array)
        abs_filename = os.path.join(BASE_DIR, filename)
        if os.path.exists(abs_filename):
            os.remove(abs_filename)
            print(f"[INFO] Removed existing file: {abs_filename}")
        img.save(abs_filename)
        print(f"[INFO] Spectator view saved to {abs_filename}")
        success = True
        del img 
    else:
        print("[ERROR] Failed to capture image from spectator view.")

    # # Clean up
    # camera.stop()
    # camera.destroy()
    # 停止监听（避免内存泄漏）
    camera.stop()
    return success

def o3d_visualizer_show_coperception(vis, count, point_cloud, predict_bbx_tensor, gt_box_tensor, show_predict, show_gt, objects, take_screenshot=False, transparent_bg=False, vid=None):
    opt = vis.get_render_option()   

    if transparent_bg:
        opt = vis.get_render_option()
        opt.background_color = (1.0, 0.0, 1.0)  # 亮粉色 (R=1, G=0, B=1)
        opt.point_size = 2.0
    else:
        opt = vis.get_render_option()
        opt.background_color = (0.0, 0.0, 0.0)  # 黑色 (R=0, G=0, B=0)
        opt.point_size = 1.0

    # if count == 2:
    vis.add_geometry(point_cloud)

    # opt.line_width = 2.0        # 全局线宽
    if show_gt:
        if gt_box_tensor is not None:
            oabbs_gt = bbx2lineset_expand(
                gt_box_tensor,
                color=(0, 1, 0),
                expand=0.10
            )
            for g in oabbs_gt:
                vis.add_geometry(g)

    # opt.line_width = 1.0       # 全局线宽
    if show_predict:
        if predict_bbx_tensor is not None:
            oabbs_pred = bbx2lineset_expand(
                predict_bbx_tensor,
                color=(1, 0, 0),
                expand=0.0
            )
            for p in oabbs_pred:
                vis.add_geometry(p)

    vis.update_geometry(point_cloud)

    for key, object_list in objects.items():
        if key != 'vehicles':
            continue
        for o in object_list:
            aabb = o.o3d_bbx
            vis.add_geometry(aabb)

    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    time.sleep(0.001)

    if take_screenshot:  # and count == 2:
        path = 'visualization_output/visualize.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vis.capture_screen_image(path)
        if transparent_bg:
            final_path = 'visualization_output/visualize_transparent.png'
            if vid:
                final_path = f'visualization_output/visualize_transparent_{vid}.png'
            make_background_transparent(path, final_path, bg_color=(255, 0, 255))
            print(f"[INFO] Transparent image saved to {final_path}")

    for key, object_list in objects.items():
        if key != 'vehicles':
            continue
        for o in object_list:
            aabb = o.o3d_bbx
            vis.remove_geometry(aabb)

    if show_predict:
        if predict_bbx_tensor is not None:
            # remove the prediction bbx drawing
            for p in oabbs_pred:
                vis.remove_geometry(p)

    if show_gt:
        if gt_box_tensor is not None:
            for g in oabbs_gt:
                vis.remove_geometry(g)


def o3d_camera_lidar_fusion(objects,
                            yolo_bbx,
                            lidar_3d,
                            projected_lidar,
                            lidar_sensor):
    """
    Utilize the 3D lidar points to extend the 2D bounding box
    from camera to 3D bounding box under world coordinates.

    Parameters
    ----------
    objects : dict
        The dictionary contains all object detection results.

    yolo_bbx : torch.Tensor
        Object detection bounding box at current photo from yolov5,
        shape (n, 5)->(n, [x1, y1, x2, y2, label])

    lidar_3d : np.ndarray
        Raw 3D lidar points in lidar coordinate system.

    projected_lidar : np.ndarray
        3D lidar points projected to the camera space.

    lidar_sensor : carla.sensor
        The lidar sensor.

    Returns
    -------
    objects : dict
        The update object dictionary that contains 3d bounding boxes.
    """

    # convert torch tensor to numpy array first
    if yolo_bbx.is_cuda:
        yolo_bbx = yolo_bbx.cpu().detach().numpy()
    else:
        yolo_bbx = yolo_bbx.detach().numpy()

    for i in range(yolo_bbx.shape[0]):
        detection = yolo_bbx[i]
        # 2d bbx coordinates
        x1, y1, x2, y2 = int(detection[0]), int(detection[1]),\
            int(detection[2]), int(detection[3])
        label = int(detection[5])

        # choose the lidar points in the 2d yolo bounding box
        points_in_bbx = \
            (projected_lidar[:, 0] > x1) & (projected_lidar[:, 0] < x2) & \
            (projected_lidar[:, 1] > y1) & (projected_lidar[:, 1] < y2) & \
            (projected_lidar[:, 2] > 0.0)
        # ignore intensity channel
        select_points = lidar_3d[points_in_bbx][:, :-1]

        if select_points.shape[0] == 0:
            continue

        # filter out the outlier
        x_common = mode(np.array(np.abs(select_points[:, 0]),
                                 dtype=np.int), axis=0)[0][0]
        y_common = mode(np.array(np.abs(select_points[:, 1]),
                                 dtype=np.int), axis=0)[0][0]
        points_inlier = (np.abs(select_points[:, 0]) > x_common - 3) & \
                        (np.abs(select_points[:, 0]) < x_common + 3) & \
                        (np.abs(select_points[:, 1]) > y_common - 3) & \
                        (np.abs(select_points[:, 1]) < y_common + 3)
        select_points = select_points[points_inlier]

        if select_points.shape[0] < 2:
            continue

        # to visualize 3d lidar points in o3d visualizer, we need to
        # revert the x coordinates
        select_points[:, :1] = -select_points[:, :1]

        # create o3d.PointCloud object
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(select_points)
        # add o3d bounding box
        aabb = o3d_pointcloud.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)

        # get the eight corner of the bounding boxes.
        corner = np.asarray(aabb.get_box_points())
        # covert back to unreal coordinate
        corner[:, :1] = -corner[:, :1]
        corner = corner.transpose()
        # extend (3, 8) to (4, 8) for homogenous transformation
        corner = np.r_[corner, [np.ones(corner.shape[1])]]
        # project to world reference
        corner = st.sensor_to_world(corner, lidar_sensor.get_transform())
        corner = corner.transpose()[:, :3]

        if is_vehicle_cococlass(label):
            obstacle_vehicle = ObstacleVehicle(corner, aabb)
            if 'vehicles' in objects:
                objects['vehicles'].append(obstacle_vehicle)
            else:
                objects['vehicles'] = [obstacle_vehicle]
        # todo: refine the category
        # we regard or other obstacle rather than vehicle as static class
        else:
            static_obstacle = StaticObstacle(corner, aabb)
            if 'static' in objects:
                objects['static'].append(static_obstacle)
            else:
                objects['static'] = [static_obstacle]

    return objects


def array_to_aabb(bb_corner):
    min_corner = np.min(bb_corner, axis=0)
    max_corner = np.max(bb_corner, axis=0)

    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corner,
                                               max_bound=max_corner)
    return aabb

def array_to_aabb_list(bounding_boxes_array, debug=False, point_cloud=None):
    aabb_list = []

    for bb_corners in bounding_boxes_array:
        # when we really run the simulation, the y axis needs to be flipped
        # for visualization purpose
        if not debug:
            tmp = bb_corners.copy()
            tmp[:, 0] = -tmp[:, 0]
            aabb = array_to_aabb(tmp)
        else:
            aabb = array_to_aabb(bb_corners)
        aabb_list.append(aabb)

    # if debug:
    #     visualize_bbx_o3d(aabb_list, point_cloud)
    return aabb_list


def is_ego_bbox(bbox):
    # filter out the points that are -3 < x < 3 and -1.5 < y < 1.5
    mask = np.logical_and(np.logical_and(bbox[:, 0] > -2.5, bbox[:, 0] < 2.5),
                          np.logical_and(bbox[:, 1] > -1.5, bbox[:, 1] < 1.5))
    return mask.any()


def o3d_predict_bbox_to_object(objects, predict_box_tensor, lidar_sensor):
    """
    Prepare objects to be returned by using predicted bbox tensor.

    predict_box_tensor: opencood predicted results. N, 8, 3 in the ego coordniate.
    """
    if predict_box_tensor is None:
        return objects

    # if predict_box_tensor.is_cuda:
    #     predict_box_tensor = predict_box_tensor.cpu().detach().numpy()
    # else:
    #     predict_box_tensor = predict_box_tensor.detach().numpy()

    # project ego coord to world coord
    # corners = st.sensor_to_world(predict_box_tensor, lidar_sensor.get_transform())

    # aabb_pred = bbx2aabb(predict_box_tensor, order='hwl')
    # aabb_list = array_to_aabb_list(predict_box_tensor)
    oabb_List = bbx2oabb(predict_box_tensor, color=(1, 0, 0))

    if predict_box_tensor.is_cuda:
        predict_box_tensor = predict_box_tensor.cpu().detach().numpy()
    else:
        predict_box_tensor = predict_box_tensor.detach().numpy()

    for i in range(len(oabb_List)):
        corner, bbox_aabb = predict_box_tensor[i], oabb_List[i]

        # remove ego bbox
        if is_ego_bbox(corner): continue

        # project ego coord to world coord
        # covert back to unreal coordinate
        # corner[:, :1] = -corner[:, :1]
        corner = corner.transpose()
        # extend (3, 8) to (4, 8) for homogenous transformation
        corner = np.r_[corner, [np.ones(corner.shape[1])]]
        # project to world reference
        corner = st.sensor_to_world(corner, lidar_sensor.get_transform())
        corner = corner.transpose()[:, :3]

        obstacle_vehicle = ObstacleVehicle(corner, bbox_aabb)
        if 'vehicles' in objects:
            objects['vehicles'].append(obstacle_vehicle)
        else:
            objects['vehicles'] = [obstacle_vehicle]

    return objects


