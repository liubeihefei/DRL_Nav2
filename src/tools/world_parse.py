import xml.etree.ElementTree as ET
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import colorsys


@dataclass
class ObjectInfo:
    """物体信息"""
    name: str
    position: Tuple[float, float]  # (x, y)
    corners_2d: List[Tuple[float, float]]  # 四个角点 (x, y)


class WorldParser:
    """从mesh文件计算实际外接框"""
    
    def __init__(self, gazebo_models_path: str):
        """
        Args:
            gazebo_models_path: Gazebo模型库路径，例如 "/home/user/.gazebo/models"
        """
        self.gazebo_models_path = gazebo_models_path
        
    def parse_world_file(self, world_file_path: str) -> List[ObjectInfo]:
        """解析世界文件，返回物体位置和实际外接框"""
        tree = ET.parse(world_file_path)
        root = tree.getroot()
        
        # 首先收集所有模型的完整定义（几何信息）
        model_definitions = self._collect_model_definitions(root)
        
        # 然后收集state中的实时位置和旋转
        state_poses = self._collect_state_poses(root)
        
        # 合并信息
        objects = []
        for model_name, model_info in model_definitions.items():
            # 获取位置和旋转：优先从state中取
            if model_name in state_poses:
                position, state_rotation = state_poses[model_name]
                print(f"\n处理物体: {model_name}")
                print(f"  state位置: ({position[0]:.3f}, {position[1]:.3f})")
                print(f"  state旋转: {state_rotation[2]:.3f} rad ({np.degrees(state_rotation[2]):.1f}°)")
                
                # 计算内部旋转（来自collision/visual）
                internal_rotation = self._compute_internal_rotation(model_info['element'])
                print(f"  内部旋转: {internal_rotation:.3f} rad ({np.degrees(internal_rotation):.1f}°)")
                
                # 总旋转 = state中的旋转 + 内部旋转
                final_rotation = state_rotation[2] + internal_rotation
                # final_rotation = state_rotation[2]
                print(f"  总旋转: {final_rotation:.3f} rad ({np.degrees(final_rotation):.1f}°)")
            else:
                position, rotation = model_info['pose']
                # 没有state数据时，才需要计算完整的总旋转
                final_rotation = self._compute_total_rotation(model_info['element'])
                print(f"\n处理物体: {model_name}")
                print(f"  使用定义位置: ({position[0]:.3f}, {position[1]:.3f})")
                print(f"  总旋转: {final_rotation:.3f} rad ({np.degrees(final_rotation):.1f}°)")
            
            # 规范化到 [-pi, pi] 范围
            final_rotation = np.arctan2(np.sin(final_rotation), np.cos(final_rotation))
            
            # 计算实际外接框
            corners = self._compute_bbox_from_mesh(
                model_info['mesh_file'], 
                model_info['scale'],
                (position[0], position[1]),  # 只取xy
                final_rotation
            )
            
            if corners:
                objects.append(ObjectInfo(
                    name=model_name,
                    position=(position[0], position[1]),
                    corners_2d=corners
                ))
        
        return objects

    def save_objects(self, objects: List[ObjectInfo], filepath: str):
        """保存物体信息到文件"""
        data = []
        for obj in objects:
            data.append({
                'name': obj.name,
                'position': [obj.position[0], obj.position[1]],
                'corners_2d': [[x, y] for x, y in obj.corners_2d]
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"已保存 {len(objects)} 个物体到 {filepath}")
    
    def load_objects(self, filepath: str) -> List[ObjectInfo]:
        """从文件加载物体信息"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        objects = []
        for item in data:
            obj = ObjectInfo(
                name=item['name'],
                position=(item['position'][0], item['position'][1]),
                corners_2d=[(x, y) for x, y in item['corners_2d']]
            )
            objects.append(obj)
        
        print(f"已加载 {len(objects)} 个物体从 {filepath}")
        return objects

    def _compute_internal_rotation(self, model: ET.Element) -> float:
        """
        计算模型内部的固定旋转（来自collision/visual）
        不包括模型本身的pose
        """
        internal_yaw = 0.0
        
        # 查找所有的link
        for link in model.findall('.//link'):
            # 查找link下的collision和visual
            for geom_type in ['collision']:
                for geom in link.findall(geom_type):
                    internal_yaw += self._get_pose_rotation(geom)
        
        return internal_yaw
    
    def _collect_model_definitions(self, root: ET.Element) -> Dict:
        """
        收集所有模型的完整定义（从<model>标签）
        包含几何信息、缩放等
        """
        model_defs = {}
        
        for model in root.findall('.//model'):
            model_name = model.get('name')
            
            # 跳过地面和墙体
            if model_name in ['ground_plane', 'wall_100x100']:
                continue
            
            # 获取模型自身的pose
            pose_elem = model.find('pose')
            if pose_elem is not None and pose_elem.text:
                pose_values = list(map(float, pose_elem.text.strip().split()))
                # 补齐到6个值 [x, y, z, roll, pitch, yaw]
                while len(pose_values) < 6:
                    pose_values.append(0.0)
                position = (pose_values[0], pose_values[1], pose_values[2])
                rotation = (pose_values[3], pose_values[4], pose_values[5])
            else:
                position = (0.0, 0.0, 0.0)
                rotation = (0.0, 0.0, 0.0)
            
            # 获取mesh信息
            mesh_info = self._get_mesh_info(model)
            if not mesh_info:
                continue
                
            mesh_file, scale = mesh_info
            
            model_defs[model_name] = {
                'pose': (position, rotation),
                'mesh_file': mesh_file,
                'scale': scale,
                'element': model
            }
        
        return model_defs
    
    def _collect_state_poses(self, root: ET.Element) -> Dict:
        """
        收集state中的实时位置
        <state>标签包含仿真时的实际位置
        """
        state_poses = {}
        
        # 查找state标签
        state = root.find('.//state')
        if state is None:
            return state_poses
        
        # 遍历state下的所有model
        for model in state.findall('.//model'):
            model_name = model.get('name')
            if not model_name:
                continue
            
            # 获取pose
            pose_elem = model.find('pose')
            if pose_elem is not None and pose_elem.text:
                pose_values = list(map(float, pose_elem.text.strip().split()))
                # 补齐到6个值
                while len(pose_values) < 6:
                    pose_values.append(0.0)
                position = (pose_values[0], pose_values[1], pose_values[2])
                rotation = (pose_values[3], pose_values[4], pose_values[5])
                state_poses[model_name] = (position, rotation)
        
        return state_poses
    
    def _get_pose_rotation(self, element: ET.Element) -> float:
        """
        从元素的pose中提取yaw旋转
        返回yaw角度（弧度）
        """
        pose_elem = element.find('pose')
        if pose_elem is not None and pose_elem.text:
            pose_values = list(map(float, pose_elem.text.strip().split()))
            if len(pose_values) > 5:
                return pose_values[5]  # 返回yaw
        return 0.0
    
    def _compute_total_rotation(self, model: ET.Element) -> float:
        """
        计算模型的总旋转角度（yaw）
        包括：模型本身的旋转 + link的旋转 + collision/visual的旋转
        """
        total_yaw = 0.0
        
        # 1. 模型本身的旋转
        total_yaw += self._get_pose_rotation(model)
        
        # 2. 查找所有的link
        for link in model.findall('.//link'):
            total_yaw += self._get_pose_rotation(link)
            
            # 3. 查找link下的collision和visual
            for geom_type in ['collision', 'visual']:
                for geom in link.findall(geom_type):
                    total_yaw += self._get_pose_rotation(geom)
        
        # 规范化到 [-pi, pi] 范围
        total_yaw = np.arctan2(np.sin(total_yaw), np.cos(total_yaw))
        
        return total_yaw
    
    def _get_mesh_info(self, model: ET.Element) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        """获取mesh文件路径和缩放"""
        # 查找collision或visual下的mesh
        for geom_type in ['.//collision', './/visual']:
            elem = model.find(f"{geom_type}//mesh")
            if elem is not None:
                uri = elem.find('uri')
                if uri is not None and uri.text:
                    mesh_file = uri.text.strip()
                    
                    # 获取缩放
                    scale = (1.0, 1.0, 1.0)
                    scale_elem = elem.find('scale')
                    if scale_elem is not None:
                        scale = tuple(map(float, scale_elem.text.strip().split()))
                    
                    return mesh_file, scale
        
        return None
    
    def _resolve_mesh_path(self, mesh_file: str) -> Optional[str]:
        """解析mesh文件的完整路径"""
        if mesh_file.startswith('model://'):
            # model://ambulance/meshes/ambulance.obj -> ambulance/meshes/ambulance.obj
            rel_path = mesh_file[8:]
            # 尝试两种路径
            path1 = os.path.join(self.gazebo_models_path, rel_path)
            if os.path.exists(path1):
                return path1
            
            # 也可能是直接在当前目录
            path2 = os.path.join(os.path.dirname(__file__), rel_path)
            if os.path.exists(path2):
                return path2
                
            return rel_path
        return mesh_file
    
    def _get_unit_scale(self, mesh_file: str) -> float:
        """
        从mesh文件获取单位缩放因子
        返回: 1单位 = ? 米
        """
        # 默认单位是米
        default_scale = 1.0
        
        try:
            if mesh_file.lower().endswith('.dae'):
                # 解析COLLADA文件获取单位信息
                tree = ET.parse(mesh_file)
                root = tree.getroot()
                
                # COLLADA命名空间
                ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
                
                # 查找unit元素
                unit_elem = root.find('.//c:unit', ns)
                if unit_elem is not None and 'meter' in unit_elem.attrib:
                    meter_value = float(unit_elem.attrib['meter'])
                    print(f"  文件 {os.path.basename(mesh_file)} 单位: 1 {unit_elem.get('name', 'unit')} = {meter_value} 米")
                    return meter_value
                else:
                    # 如果没有找到unit，可能是默认米
                    print(f"  文件 {os.path.basename(mesh_file)} 使用默认单位 (米)")
                    return default_scale
                    
            elif mesh_file.lower().endswith('.obj'):
                # OBJ文件通常单位是米或没有定义单位，这里可以尝试读取MTL或其他信息
                # 但大多数Gazebo模型使用米
                print(f"  OBJ文件 {os.path.basename(mesh_file)} 使用默认单位 (米)")
                return default_scale
                
            elif mesh_file.lower().endswith('.stl'):
                # STL文件没有单位信息，通常假设为米
                print(f"  STL文件 {os.path.basename(mesh_file)} 使用默认单位 (米)")
                return default_scale
                
        except Exception as e:
            print(f"  获取单位信息时出错: {e}")
            
        return default_scale
    
    def _extract_vertices_from_mesh(self, mesh) -> np.ndarray:
        """从mesh或scene中提取顶点"""
        if hasattr(mesh, 'vertices'):
            # 如果是单个Mesh对象
            return mesh.vertices
        elif hasattr(mesh, 'geometry'):
            # 如果是Scene对象，合并所有几何体的顶点
            vertices = []
            for geom_name, geom in mesh.geometry.items():
                if hasattr(geom, 'vertices'):
                    vertices.append(geom.vertices)
            if vertices:
                return np.vstack(vertices)
        elif isinstance(mesh, dict) and 'vertices' in mesh:
            # 某些情况下可能是字典
            return mesh['vertices']
        
        raise ValueError("无法从mesh中提取顶点")
    
    def _compute_bbox_from_mesh(self, mesh_file: str, scale: Tuple[float, float, float],
                               position: Tuple[float, float], rotation: float) -> Optional[List[Tuple[float, float]]]:
        """从mesh文件计算外接框的四个角点"""
        try:
            # 获取完整路径
            full_path = self._resolve_mesh_path(mesh_file)
            if not os.path.exists(full_path):
                print(f"文件不存在: {full_path}")
                return None
            
            print(f"\n处理文件: {full_path}")
            
            # 获取文件单位缩放
            unit_scale = self._get_unit_scale(full_path)
            
            # 加载mesh，强制为mesh而不是scene
            try:
                mesh = trimesh.load(full_path, force='mesh', skip_materials=True)
            except:
                # 如果force='mesh'失败，尝试普通加载然后手动提取
                mesh = trimesh.load(full_path, skip_materials=True)
            
            # 提取顶点
            try:
                vertices = self._extract_vertices_from_mesh(mesh)
                print(f"  顶点数量: {len(vertices)}")
            except:
                print(f"  无法从 {mesh_file} 提取顶点，尝试使用包围盒近似")
                # 如果无法提取顶点，尝试获取mesh的包围盒
                if hasattr(mesh, 'bounds'):
                    bounds = mesh.bounds
                    # 创建包围盒的8个顶点
                    vertices = np.array([
                        [bounds[0][0], bounds[0][1], bounds[0][2]],
                        [bounds[1][0], bounds[0][1], bounds[0][2]],
                        [bounds[1][0], bounds[1][1], bounds[0][2]],
                        [bounds[0][0], bounds[1][1], bounds[0][2]],
                        [bounds[0][0], bounds[0][1], bounds[1][2]],
                        [bounds[1][0], bounds[0][1], bounds[1][2]],
                        [bounds[1][0], bounds[1][1], bounds[1][2]],
                        [bounds[0][0], bounds[1][1], bounds[1][2]]
                    ])
                else:
                    return None
            
            if len(vertices) == 0:
                return None
            
            # 显示原始范围（在模型坐标系中）
            orig_min = np.min(vertices, axis=0)
            orig_max = np.max(vertices, axis=0)
            print(f"  原始范围: X[{orig_min[0]:.3f}, {orig_max[0]:.3f}], "
                  f"Y[{orig_min[1]:.3f}, {orig_max[1]:.3f}], "
                  f"Z[{orig_min[2]:.3f}, {orig_max[2]:.3f}]")
            
            # 应用单位缩放（先将模型单位转换为米）
            vertices = vertices * unit_scale
            print(f"  单位缩放后: 因子={unit_scale}")
            
            # 应用模型文件中的缩放（Gazebo的<scale>标签）
            vertices = vertices * scale
            print(f"  Gazebo缩放: {scale}")
            
            # 只取xy平面上的点
            points_2d = vertices[:, :2]
            
            # 计算旋转矩阵
            cos_a = np.cos(rotation)
            sin_a = np.sin(rotation)
            
            # 应用旋转和平移
            rotated_x = points_2d[:, 0] * cos_a - points_2d[:, 1] * sin_a + position[0]
            rotated_y = points_2d[:, 0] * sin_a + points_2d[:, 1] * cos_a + position[1]
            
            transformed_points = np.column_stack([rotated_x, rotated_y])
            
            # 计算最小和最大坐标
            min_x, min_y = np.min(transformed_points, axis=0)
            max_x, max_y = np.max(transformed_points, axis=0)
            
            print(f"  最终范围: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")
            
            # 返回四个角点 [左下, 右下, 右上, 左上]
            corners = [
                (float(min_x), float(min_y)),
                (float(max_x), float(min_y)),
                (float(max_x), float(max_y)),
                (float(min_x), float(max_y))
            ]
            
            return corners
            
        except Exception as e:
            print(f"处理mesh文件 {mesh_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def plot_objects(objects: List[ObjectInfo], save_path: Optional[str] = None):
    """
    将所有物体的外接框画在100x100的图上（范围-50到50）
    
    Args:
        objects: 物体信息列表
        save_path: 如果指定，保存图片到该路径
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # # 设置坐标轴范围
    # ax.set_xlim(-50, 50)
    # ax.set_ylim(-50, 50)

    # 设置坐标轴范围
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Object Bounding Boxes in World (-50 to 50)', fontsize=14)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴等比例
    ax.set_aspect('equal')
    
    # 为每个物体生成不同的颜色
    colors = []
    for i in range(len(objects)):
        # 使用HSV颜色空间生成不同的颜色
        hue = i / len(objects) if len(objects) > 0 else 0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)
    
    # 绘制每个物体的外接框
    for idx, obj in enumerate(objects):
        corners = obj.corners_2d
        
        # 创建矩形
        min_x, min_y = corners[0]
        max_x, max_y = corners[2]
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 创建矩形patch
        rect = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=2,
            edgecolor=colors[idx],
            facecolor='none',
            alpha=0.8,
            label=f"{obj.name}"
        )
        
        # 添加矩形到图形
        ax.add_patch(rect)
    
    # 添加图例
    if len(objects) <= 20:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    else:
        ax.text(1.05, 0.5, f'Total objects: {len(objects)}',
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='center',
               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # 添加坐标轴原点标记
    ax.plot(0, 0, 'ro', markersize=5, label='Origin')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def print_statistics(objects: List[ObjectInfo]):
    """打印物体统计信息"""
    print("\n" + "="*60)
    print("物体统计信息")
    print("="*60)
    
    print(f"总物体数: {len(objects)}")
    
    # 计算尺寸统计
    if objects:
        sizes = []
        for obj in objects:
            corners = obj.corners_2d
            min_x = min(p[0] for p in corners)
            max_x = max(p[0] for p in corners)
            min_y = min(p[1] for p in corners)
            max_y = max(p[1] for p in corners)
            width = max_x - min_x
            height = max_y - min_y
            sizes.append((width, height))
        
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        print(f"\n尺寸统计:")
        print(f"  宽度 - 最小: {min(widths):.2f}, 最大: {max(widths):.2f}, 平均: {np.mean(widths):.2f}")
        print(f"  高度 - 最小: {min(heights):.2f}, 最大: {max(heights):.2f}, 平均: {np.mean(heights):.2f}")


# 使用示例
def main():
    # 设置你的Gazebo模型路径
    gazebo_path = "/home/horsefly/.gazebo/models"
    
    parser = WorldParser(gazebo_path)
    
    # 解析世界文件
    world_file = "/home/horsefly/下载/40by40.model"
    objects = parser.parse_world_file(world_file)
    
    # 保存到文件
    save_file = "objects.json"
    parser.save_objects(objects, save_file)
    
    # 从文件加载
    loaded_objects = parser.load_objects(save_file)
    
    # 输出结果
    print(f"\n找到 {len(objects)} 个物体\n")
    
    for obj in objects:
        print(f"物体: {obj.name}")
        print(f"  位置: ({obj.position[0]:.3f}, {obj.position[1]:.3f})")
        print(f"  外接框:")
        for i, (x, y) in enumerate(obj.corners_2d):
            print(f"    p{i+1}: ({x:.3f}, {y:.3f})")
        
        # 计算尺寸
        if obj.corners_2d:
            min_x = min(p[0] for p in obj.corners_2d)
            max_x = max(p[0] for p in obj.corners_2d)
            min_y = min(p[1] for p in obj.corners_2d)
            max_y = max(p[1] for p in obj.corners_2d)
            width = max_x - min_x
            height = max_y - min_y
            print(f"  尺寸: {width:.3f} x {height:.3f}")
        print()
    
    # 绘制所有物体的外接框
    print("\n" + "="*60)
    print("开始绘制可视化图形...")
    print("="*60)
    
    # 保存图片到文件
    save_path = "object_bboxes.png"
    plot_objects(objects, save_path)
    
    # 打印统计信息
    print_statistics(objects)


if __name__ == "__main__":
    main()