
import cv2
import numpy as np
import json
import os
from Open3D import *
import yaml

from dataclasses import dataclass

@dataclass
class Params:
    depth_scale = 0.001
    plane_fit_threshold = 0.005
    ransac_max_iterations = 1000
    ransac_min_inlier_ratio = 0.5
    converage_threshold = 0.1
    min_point_cloud_size = 50
    boundary_filter_ratio = 0.02
    excepted_tag_width_mm = 0.0
    excepted_tag_height_mm = 0.0
    tag_size_ratio_threshold = 0.5
    correct_orientation = True
    apply_right_camera_rotation = False
    min_mask_pixel_count = 100
    pose_axis_length = 0.1
    seed =7
    object_label = "tag_code"
    object_confidence = 0.95

@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

def align_depth_to_intrinsics(depth_u16,intr):
    target_shape = (intr.height,intr.width)
    aligned = cv2.resize(depth_u16,(intr.width,intr.height),interpolation=cv2.INTER_NEAREST)
    return aligned,f"resized _to_intrinsics:{intr.width}*{intr.height}"
    

def load_intrinsics_from_yaml(yaml_path):
    with open(yaml_path,"r",encoding="utf-8") as f:
        data = yaml.safe_load(f)

    node = data.get("intrinsics")
    return Intrinsics(
        width=int(node.get("width",0)),
        height=int(node.get("height",0)),
        fx = float(node["fx"]),
        fy = float(node["fy"]),
        cx = float(node["cx"]),
        cy = float(node["cy"]),
    ),float(data.get("depth_scale"))

def mask_from_bbox(image_shape,bbox):
    h_img,w_img = image_shape
    x,y,w,h = bbox
    x0 = max(0,x)
    y0 = max(0,y)
    x1 = min(w_img,x+w)
    y1 = min(h_img,y+h)
    mask = np.zeros((h_img,w_img),dtype=np.uint8)
    mask[y0:y1,x0:x1] = 255
    return mask

def mask_to_points3d(mask,depth_u16,intr,depth_scale):
    ys,xs = np.nonzero(mask)
    raw_depth = depth_u16[ys,xs].astype(np.float64)
    z = raw_depth * depth_scale
    valid = (raw_depth > 0) & (z > 0.1) & (z< 10.0) & np.isfinite(z)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid]
    x = (xs - intr.cx) * z / intr.fx
    y = (ys - intr.cy) * z / intr.fy
    return np.column_stack([x,y,z]).astype(np.float64)

def fit_plane_ransac(points,params):
    rng = np.random.default_rng(params.seed)
    min_inliers = int(points.shape[0] * params.ransac_min_inlier_ratio)

    best_mask = None
    best_normal = None
    best_count = 0

    for _ in range(params.ransac_max_iterations):
        idx = rng.choice(points.shape[0],size=3,replace=False)
        p1,p2,p3 = points[idx]
        normal = np.cross(p2-p1,p3-p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        distances = np.abs((points - p1)@normal)
        inlier_mask = distances <= params.plane_fit_threshold
        count = int(np.count_nonzero(inlier_mask))
        if count > best_count:
            best_count = count
            best_mask = inlier_mask
            best_normal = normal

    if best_mask is None or best_normal is None or best_count < min_inliers:
        print(f"RANSAC failed:inliers {best_count}/{points.shape[0]},need {min_inliers}")
        exit()
    inliers = points[best_mask]
    center = inliers.mean(axis=0)
    distances = np.abs((inliers - center) @ best_normal)
    avg_distance = float(distances.mean()) if len(distances) else 0.0
    return center,best_normal,inliers,avg_distance

# 将法向量指向相机中心
def correct_plane_normal_direction(center,normal):
    to_origin = - center
    if float(normal @ to_origin) < 0.0:
        return -normal
    return normal

def rotation_x_pi():
    return np.array([[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]])

def rotation_z_pi():
    return np.array([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]])

def correct_pose_orientation(rotation):
    pose_z_axis = rotation[:,2]
    pose_x_axis = rotation[:,0]
    camera_z = np.array([0.0,0.0,1.0])
    if float(pose_z_axis @ camera_z) > 0.0:
        rotation = rotation @ rotation_x_pi()
        pose_x_axis = rotation[:,0]
    if float(pose_x_axis @ np.array([0.0,1.0,0.0])) > 0.0:
        rotation = rotation @ rotation_z_pi()
    return rotation

def compute_pose_from_plane(center,normal,inliers,params):
    # 这段代码先用normal构造一个平面坐标系，把3D内点压到平面并转成2D,然后用最小外接矩形和PCA找目标中心与主方向，最后再转回3D位姿
    plane_z = normal.astype(np.float64)
    plane_z /= np.linalg.norm(plane_z)

    if abs(float(plane_z[0])) < 0.9:
        plane_x = np.array([1.0,0.0,0.0])
    else:
        plane_x = np.array([0.0,1.0,0.0])

    # 这个是重点,用plane_x减去plane_x在plane_z上的投影，两个向量相减就会得到垂直于plane_z的向量
    plane_x = plane_x - float(plane_x @ plane_z) * plane_z
    plane_x /= np.linalg.norm(plane_x)
    plane_y = np.cross(plane_x,plane_z)
    show_plane_z_vector(center,plane_z,plane_x,plane_y)

    # 把3D平面的投影到拟合出来的平面上，然后再把这些3D点转换成平面自己的2D坐标
    # 每个点到平面中心的向量,从中点出发,指向inliers的向量
    rel = inliers - center
    # 这个点离拟合平面有多远
    dist_to_plane = rel @ plane_z
    # 把点沿着法向量方向压回平面上，得到是inliers投影到Z平面上的点,向量平行的原理
    projected = inliers - dist_to_plane[:,None] * plane_z[None,:]
    # 投影后的点，再转成相对于center的向量,中心指向投影点的方向
    rel_projected = projected - center
    # 把3D点转成2D平面坐标,点在X方向上的投影和Y方向上的投影
    points_2d = np.column_stack([rel_projected @ plane_x,rel_projected @ plane_y]).astype(np.float32)
    show_points_2d(points_2d)

    if points_2d.shape[0] < 3:
        print(f"点数小于3个")
        exit()

    min_rect = cv2.minAreaRect(points_2d.reshape(-1,1,2))
    rect_center_2d = np.array(min_rect[0],dtype=np.float64)
    rect_width = float(min_rect[1][0])
    rect_height = float(min_rect[1][1])
    # 目标平面矩形的中心点转化为相机坐标系下的点，原理是向量平行
    translation = center + rect_center_2d[0]*plane_x + rect_center_2d[1]*plane_y
    show_translation_relation(center,translation,rect_center_2d,plane_x,plane_y)

    # 每个点都变成了相对于平均中心的偏移
    centered_2d = points_2d.astype(np.float64) - points_2d.mean(axis=0)
    # 对协方差矩阵做特征值分解
    covariance = (centered_2d.T @ centered_2d) / max(points_2d.shape[0]-1,1)
    eigvals,eigvecs = np.linalg.eigh(covariance)
    del eigvals

    # 在PCA中,较大的特征值对应的方向 = 点云分布最长的方向 = 主方向
    #        较小特征值对应的方向 = 点云分布较短的方向 = 次方向

    primary_2d = eigvecs[:,1]   # 点云最长的方向
    secondary_2d = eigvecs[:,0] # 点云较短的方向
    primary_direction = np.array([primary_2d[0],primary_2d[1],0.0])  # 把 2D向量扩展成 3D向量
    secondary_direction = np.array([secondary_2d[0],secondary_2d[1],0.0])

    if primary_direction[0] < 0.0:
        primary_direction =-primary_direction
    if np.cross(primary_direction,secondary_direction)[2] < 0.0:
        secondary_direction = -secondary_direction

    # 将局部坐标系下的主方向转换为相机坐标系下的主方向
    world_primary = primary_direction[0] * plane_x + primary_direction[1] * plane_y
    show_primary_direction_relation(primary_direction,world_primary,plane_x,plane_y)

    final_z = plane_z /np.linalg.norm(plane_z)
    # 目标物体的Y轴 = 平面内点云最长方向
    final_y = world_primary / np.linalg.norm(world_primary)
    final_x = np.cross(final_y,final_z)

    final_x /= np.linalg.norm(final_x)
    # 防止有误差，再重复算一次Y
    final_y = np.cross(final_z,final_x)
    final_y /= np.linalg.norm(final_y)

    # 将三根轴按列拼成旋转矩阵
    rotation = np.column_stack([final_x,final_y,final_z])
    if params.correct_orientation:
        rotation = correct_pose_orientation(rotation)
    if params.apply_right_camera_rotation:
        rotation = rotation @ rotation_z_pi()

    return translation,rotation,rect_width,rect_height,points_2d

def project_point(point,intr):
    z = max(float(point[2]),1e-9)
    u = int(round(float(point[0]) * intr.fx /z + intr.cx))
    v = int(round(float(point[1]) * intr.fy /z + intr.cy))
    return u,v

def draw_debug(src_bgr,mask,bbox,translation,rotation,intr,pose_axis_length,label,confidence):
    out = src_bgr.copy()
    x,y,w,h = bbox
    cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
    text =f"{label} {int(confidence * 100)}%"
    text_size,_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    text_y = max(y-5,text_size[1]+2)
    cv2.putText(out,text,(x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    draw_mask = mask
    draw_mask = cv2.resize(draw_mask,(out.shape[1],out.shape[0]),interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros_like(out)
    color_mask[draw_mask >0] = (0,200,0)
    out = cv2.addWeighted(out,1.0,color_mask,0.3,0)

    origin = project_point(translation,intr)
    axes = [
        (rotation[:,0],(0,0,255),"x"),
        (rotation[:,1],(0,255,0),"y"),
        (rotation[:,2],(255,0,0),"z")
    ]
    cv2.circle(out,origin,4,(0,255,255),-1)
    for axis,color,name in axes:
        end = project_point(translation + axis * pose_axis_length,intr)
        cv2.arrowedLine(out,origin,end,color,2,tipLength=0.2)
        cv2.putText(out,name,end,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
    return out

def show_points_2d(points_2d,output_path=None,window_name="points_2d"):
    if points_2d.shape[0] < 1:
        print("points_2d为空，无法显示")
        return

    pts = points_2d.astype(np.float64)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy,1e-9)
    canvas_size = 720
    margin = 60
    scale = min((canvas_size - margin * 2) / span[0],(canvas_size - margin * 2) / span[1])

    img = np.full((canvas_size,canvas_size,3),255,dtype=np.uint8)

    def to_pixel(point):
        x = int(round((point[0] - min_xy[0]) * scale + margin))
        y = int(round(canvas_size - ((point[1] - min_xy[1]) * scale + margin)))
        return x,y

    origin = to_pixel(np.array([0.0,0.0]))
    cv2.line(img,(0,origin[1]),(canvas_size,origin[1]),(220,220,220),1)
    cv2.line(img,(origin[0],0),(origin[0],canvas_size),(220,220,220),1)
    cv2.circle(img,origin,4,(0,0,255),-1)

    for point in pts:
        cv2.circle(img,to_pixel(point),2,(255,120,0),-1)

    if pts.shape[0] >= 3:
        min_rect = cv2.minAreaRect(points_2d.reshape(-1,1,2))
        box = cv2.boxPoints(min_rect)
        box_pixels = np.array([to_pixel(point) for point in box],dtype=np.int32)
        cv2.polylines(img,[box_pixels],True,(0,180,0),2,cv2.LINE_AA)
        rect_center = to_pixel(np.array(min_rect[0],dtype=np.float64))
        cv2.circle(img,rect_center,5,(0,0,255),-1)

    cv2.putText(img,"points_2d: plane local X/Y",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img,"blue: points  green: minAreaRect  red: origin/center",(20,canvas_size-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,80,80),1,cv2.LINE_AA)

    print(f"points_2d shape: {points_2d.shape}")
    print(f"points_2d x范围: {min_xy[0]:.6f} 到 {max_xy[0]:.6f}")
    print(f"points_2d y范围: {min_xy[1]:.6f} 到 {max_xy[1]:.6f}")
    print("points_2d前10个点:")
    print(points_2d[:10])

    if output_path is not None:
        cv2.imwrite(output_path,img)
        print(f"points_2d显示图已保存: {output_path}")

    cv2.imshow(window_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def show_translation_relation(center,translation,rect_center_2d,plane_x,plane_y):
    u = float(rect_center_2d[0])
    v = float(rect_center_2d[1])
    x_offset = u * plane_x
    y_offset = v * plane_y
    x_tip = center + x_offset

    print("translation向量关系:")
    print(f"center: {center}")
    print(f"rect_center_2d: [{u:.6f}, {v:.6f}]")
    print(f"plane_x: {plane_x}")
    print(f"plane_y: {plane_y}")
    print(f"u * plane_x: {x_offset}")
    print(f"v * plane_y: {y_offset}")
    print(f"translation = center + u * plane_x + v * plane_y: {translation}")

    show_translation_relation_2d(center,translation,rect_center_2d,plane_x,plane_y)

def show_translation_relation_2d(center,translation,rect_center_2d,plane_x,plane_y,window_name="translation_relation"):
    u = float(rect_center_2d[0])
    v = float(rect_center_2d[1])
    canvas_size = 720
    margin = 90
    max_abs = max(abs(u),abs(v),0.05)
    scale = (canvas_size / 2 - margin) / max_abs
    origin = np.array([canvas_size // 2,canvas_size // 2],dtype=np.float64)

    def to_pixel(point):
        x = int(round(origin[0] + point[0] * scale))
        y = int(round(origin[1] - point[1] * scale))
        return x,y

    img = np.full((canvas_size,canvas_size,3),255,dtype=np.uint8)
    cv2.line(img,(margin,canvas_size // 2),(canvas_size - margin,canvas_size // 2),(220,220,220),1)
    cv2.line(img,(canvas_size // 2,margin),(canvas_size // 2,canvas_size - margin),(220,220,220),1)

    center_px = to_pixel(np.array([0.0,0.0]))
    x_tip_px = to_pixel(np.array([u,0.0]))
    translation_px = to_pixel(np.array([u,v]))
    plane_x_px = to_pixel(np.array([max_abs * 0.7,0.0]))
    plane_y_px = to_pixel(np.array([0.0,max_abs * 0.7]))

    cv2.arrowedLine(img,center_px,plane_x_px,(0,0,255),2,tipLength=0.08)
    cv2.arrowedLine(img,center_px,plane_y_px,(0,160,0),2,tipLength=0.08)
    cv2.arrowedLine(img,center_px,x_tip_px,(0,0,255),2,tipLength=0.08)
    cv2.arrowedLine(img,x_tip_px,translation_px,(0,160,0),2,tipLength=0.08)
    cv2.arrowedLine(img,center_px,translation_px,(180,0,180),2,tipLength=0.08)

    cv2.circle(img,center_px,6,(0,200,255),-1)
    cv2.circle(img,translation_px,6,(180,0,180),-1)

    cv2.putText(img,"center",(center_px[0] + 10,center_px[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,120,180),2,cv2.LINE_AA)
    cv2.putText(img,"plane_x",plane_x_px,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img,"plane_y",plane_y_px,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,150,0),2,cv2.LINE_AA)
    cv2.putText(img,f"rect_center_2d=({u:.4f},{v:.4f})",(translation_px[0] + 10,translation_px[1] + 18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,80,80),1,cv2.LINE_AA)
    cv2.putText(img,"translation",(translation_px[0] + 10,translation_px[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,0,180),2,cv2.LINE_AA)
    cv2.putText(img,"translation = center + u*plane_x + v*plane_y",(20,canvas_size - 25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(60,60,60),1,cv2.LINE_AA)

    print("一张图显示: center、translation、rect_center_2d、plane_x、plane_y")
    print(f"center(3D): {center}")
    print(f"translation(3D): {translation}")
    print(f"rect_center_2d: [{u:.6f}, {v:.6f}]")
    print(f"plane_x(3D方向): {plane_x}")
    print(f"plane_y(3D方向): {plane_y}")

    cv2.imshow(window_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def show_primary_direction_relation(primary_direction,world_primary,plane_x,plane_y,window_name="primary_direction_relation"):
    a = float(primary_direction[0])
    b = float(primary_direction[1])
    canvas_size = 720
    margin = 90
    max_abs = max(abs(a),abs(b),1.0)
    scale = (canvas_size / 2 - margin) / max_abs
    origin = np.array([canvas_size // 2,canvas_size // 2],dtype=np.float64)

    def to_pixel(point):
        x = int(round(origin[0] + point[0] * scale))
        y = int(round(origin[1] - point[1] * scale))
        return x,y

    img = np.full((canvas_size,canvas_size,3),255,dtype=np.uint8)
    center_px = to_pixel(np.array([0.0,0.0]))
    plane_x_px = to_pixel(np.array([0.85,0.0]))
    plane_y_px = to_pixel(np.array([0.0,0.85]))
    primary_px = to_pixel(np.array([a,b]))
    x_part_px = to_pixel(np.array([a,0.0]))

    cv2.line(img,(margin,canvas_size // 2),(canvas_size - margin,canvas_size // 2),(225,225,225),1)
    cv2.line(img,(canvas_size // 2,margin),(canvas_size // 2,canvas_size - margin),(225,225,225),1)

    cv2.arrowedLine(img,center_px,plane_x_px,(0,0,255),2,tipLength=0.08)
    cv2.arrowedLine(img,center_px,plane_y_px,(0,160,0),2,tipLength=0.08)
    cv2.arrowedLine(img,center_px,x_part_px,(0,0,255),2,tipLength=0.08)
    cv2.arrowedLine(img,x_part_px,primary_px,(0,160,0),2,tipLength=0.08)
    cv2.arrowedLine(img,center_px,primary_px,(255,120,0),3,tipLength=0.08)

    cv2.circle(img,center_px,6,(0,200,255),-1)
    cv2.circle(img,primary_px,6,(255,120,0),-1)

    cv2.putText(img,"plane_x",plane_x_px,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img,"plane_y",plane_y_px,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,150,0),2,cv2.LINE_AA)
    cv2.putText(img,"primary_direction",(primary_px[0] + 10,primary_px[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,120,0),2,cv2.LINE_AA)
    cv2.putText(img,f"primary_direction=({a:.4f},{b:.4f},0)",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(img,"world_primary = a*plane_x + b*plane_y",(20,canvas_size - 55),cv2.FONT_HERSHEY_SIMPLEX,0.55,(60,60,60),1,cv2.LINE_AA)
    cv2.putText(img,f"world_primary=({world_primary[0]:.4f},{world_primary[1]:.4f},{world_primary[2]:.4f})",(20,canvas_size - 25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(60,60,60),1,cv2.LINE_AA)

    print("primary_direction和world_primary向量关系:")
    print(f"primary_direction: {primary_direction}")
    print(f"plane_x: {plane_x}")
    print(f"plane_y: {plane_y}")
    print(f"world_primary = primary_direction[0] * plane_x + primary_direction[1] * plane_y: {world_primary}")

    cv2.imshow(window_name,img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def show_ransac_result(points,center,normal,inliers,normal_length=0.2):
    # 全部点：灰色
    all_pcd = points_to_pcd(points)
    all_pcd.paint_uniform_color([0.5,0.5,0.5])

    # 平面内点： 绿色
    inlier_pcd = points_to_pcd(inliers)
    inlier_pcd.paint_uniform_color([0.0,1.0,0.0])

    # 中心点：黄色小球
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1.0,0.5,1.0])

    # 法向量: 红色箭头
    normal = normal / np.linalg.norm(normal)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius = 0.005,
        cone_radius = 0.015,
        cylinder_height = normal_length * 0.8,
        cone_height = normal_length *0.2,
    )
    arrow.paint_uniform_color([1.0,0.0,0.0])

    z_axis = np.array([0.0,0.0,1.0])
    R = rotation_matrix_from_vectors(z_axis,normal)
    arrow.rotate(R,center=np.array([0.0,0.0,0.0]))
    arrow.translate(center)

    end = center + normal * normal_length
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    end_sphere.translate(end)
    end_sphere.paint_uniform_color([1.0,0.0,0.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3,origin=[0,0,0])

    o3d.visualization.draw_geometries([all_pcd,inlier_pcd,center_sphere,end_sphere,arrow,axis])

def show_plane_z_vector(center,plane_z,plane_x=None,plane_y=None,vector_length=0.2):
    plane_z = plane_z.astype(np.float64)
    plane_z /= np.linalg.norm(plane_z)
    if plane_x is not None:
        plane_x = plane_x.astype(np.float64)
        plane_x /= np.linalg.norm(plane_x)
    if plane_y is not None:
        plane_y = plane_y.astype(np.float64)
        plane_y /= np.linalg.norm(plane_y)

    start = np.asarray(center,dtype=np.float64)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    start_sphere.translate(start)
    start_sphere.paint_uniform_color([1.0,1.0,0.0])

    geometries = [start_sphere]
    geometries.extend(make_vector_geometries(start,plane_z,vector_length,[0.0,0.2,1.0]))
    if plane_x is not None:
        geometries.extend(make_vector_geometries(start,plane_x,vector_length,[1.0,0.0,0.0]))
    if plane_y is not None:
        geometries.extend(make_vector_geometries(start,plane_y,vector_length,[0.0,1.0,0.0]))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3,origin=[0,0,0])
    geometries.append(axis)
    print(f"plane_z start: {start}")
    print(f"plane_z end: {start + plane_z * vector_length}")
    if plane_x is not None:
        print(f"plane_x start: {start}")
        print(f"plane_x end: {start + plane_x * vector_length}")
    if plane_y is not None:
        print(f"plane_y start: {start}")
        print(f"plane_y end: {start + plane_y * vector_length}")
    o3d.visualization.draw_geometries(geometries)

def make_vector_geometries(start,direction,vector_length,color):
    end = start + direction * vector_length

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    end_sphere.translate(end)
    end_sphere.paint_uniform_color(color)

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([start,end])
    line.lines = o3d.utility.Vector2iVector([[0,1]])
    line.colors = o3d.utility.Vector3dVector([color])

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003,
        cone_radius=0.009,
        cylinder_height=vector_length * 0.8,
        cone_height=vector_length * 0.2,
    )
    arrow.paint_uniform_color(color)
    R = rotation_matrix_from_vectors(np.array([0.0,0.0,1.0]),direction)
    arrow.rotate(R,center=np.array([0.0,0.0,0.0]))
    arrow.translate(start)

    return [line,arrow,end_sphere]


def rotation_matrix_from_vectors(vec1,vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    v = np.cross(vec1,vec2)
    c = np.dot(vec1, vec2)
    if np.isclose(c,1.0):
        return np.eye(3)

    if np.isclose(c,-1.0):
        return np.array([
            [1.0,0.0,0.0],
            [0.0,-1.0,0.0],
            [0.0,0.0,-1.0]
        ])

    vx = np.array([
        [0.0,-v[2],v[1]],
        [v[2],0.0,-v[0]],
        [-v[1],v[0],0.0],
    ])

    R = np.eye(3) +vx +vx@vx * ((1.0-c)/(np.linalg.norm(v))**2)
    return R



def estimate_6d_pose(src_image_path,depth_image_path,result_image_path,yaml_path,params,bbox,output_dir):
    intr,params.depth_scale = load_intrinsics_from_yaml(yaml_path)
    depth = cv2.imread(depth_image_path,cv2.IMREAD_UNCHANGED)
    depth,depth_alignment = align_depth_to_intrinsics(depth,intr)

    src_bgr = cv2.imread(src_image_path,cv2.IMREAD_COLOR)
    result_bgr = cv2.imread(result_image_path,cv2.IMREAD_COLOR)

    mask = mask_from_bbox(depth.shape[:2],bbox)
    points = mask_to_points3d(mask,depth,intr,params.depth_scale)

    save_xyz(points,output_dir +"/points.txt")

    # pcd = points_to_pcd(points)
    # show_ply([pcd])

    # 法线方向和平面的三个角是什么关系
    # 法线方向的三个值是什么意思

    center,normal,inliers,avg_distance = fit_plane_ransac(points,params)
    # show_ransac_result(points,center,normal,inliers)
    normal = correct_plane_normal_direction(center,normal)

    translation,rotation,rect_w,rect_h,points_2d = compute_pose_from_plane(center,normal,inliers,params)

    rect_area_mm2 = rect_w * rect_h *1e6
    coverage_ratio = min(float(inliers.shape[0]) / rect_area_mm2,1.0) if rect_area_mm2 >0 else 0.0

    mask_path = output_dir + "/" + f"_mask.png"
    cv2.imwrite(mask_path,mask)

    # 画结果显示图
    debug = draw_debug(src_bgr,mask,bbox,translation,rotation,intr,params.pose_axis_length,params.object_label,params.object_confidence)
    debug_path = output_dir + "/" + f"_debug.jpg"
    cv2.imwrite(debug_path,debug)


if __name__ == "__main__":
    src_image_path = "/Users/jodocls/Desktop/code/my_hub/4-Open3D/data/6D-Pose/20260721_191340.072_src.png"
    depth_image_path = "/Users/jodocls/Desktop/code/my_hub/4-Open3D/data/6D-Pose/20260721_191340.073_depth.png"
    yaml_path = "/Users/jodocls/Desktop/code/my_hub/4-Open3D/data/6D-Pose/20260721_191340.076_meta.yaml"
    result_image_path = "/Users/jodocls/Desktop/code/my_hub/4-Open3D/data/6D-Pose/20260721_191340.073_result.jpg"
    output_path = "/Users/jodocls/Desktop/code/my_hub/tmp/6D位姿估计"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    params = Params()
    estimate_6d_pose(src_image_path,depth_image_path,result_image_path,yaml_path,params,[640,360,263,125],output_path)
