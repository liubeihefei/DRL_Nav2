#!/usr/bin/env python3
"""
重命名Gazebo世界文件中的障碍物为 obs1, obs2, obs3...
使用方法: python rename_obstacles.py input.world output.world
"""

import xml.etree.ElementTree as ET
import sys
import os
import re

def rename_obstacles(input_file, output_file):
    """
    将除了ground_plane和wall_100x100之外的模型重命名为obs1, obs2, obs3...
    """
    try:
        # 注册命名空间
        ET.register_namespace('', 'http://sdformat.org/schemas/root.xsd')
        
        # 解析XML文件
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        # 找到world元素
        world = root.find('world')
        if world is None:
            print("错误：找不到world元素")
            return False
        
        # 要保留原名的模型
        excluded_models = ['ground_plane', 'wall_100x100']
        
        # 收集所有模型
        models = world.findall('model')
        
        # 首先，找出需要重命名的模型
        models_to_rename = []
        for model in models:
            model_name = model.get('name', '')
            if model_name not in excluded_models:
                models_to_rename.append(model)
        
        print(f"找到 {len(models_to_rename)} 个需要重命名的模型")
        
        # 创建名称映射
        name_mapping = {}
        for i, model in enumerate(models_to_rename, 1):
            old_name = model.get('name')
            new_name = f"obs{i}"
            name_mapping[old_name] = new_name
            model.set('name', new_name)
            print(f"重命名: {old_name} -> {new_name}")
        
        # 同时更新state中的模型名称
        state = world.find('state')
        if state is not None:
            state_models = state.findall('model')
            for state_model in state_models:
                old_name = state_model.get('name')
                if old_name in name_mapping:
                    new_name = name_mapping[old_name]
                    state_model.set('name', new_name)
                    print(f"更新state: {old_name} -> {new_name}")
        
        print(f"\n重命名完成! 共重命名 {len(models_to_rename)} 个模型")
        
        # 保存文件
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"文件已保存到: {output_file}")
        
        return True
        
    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False

def main():
    """
    主函数
    """
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python rename_obstacles.py <input_file> [output_file]")
        print("\n示例:")
        print("  python rename_obstacles.py myworld.world")
        print("  python rename_obstacles.py myworld.world renamed.world")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 '{input_file}' 不存在")
        sys.exit(1)
    
    # 确定输出文件名
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # 在文件名后添加 "_renamed"
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_renamed{ext}"
    
    print(f"正在处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    # 执行重命名
    success = rename_obstacles(input_file, output_file)
    
    if success:
        print("\n处理完成！")
    else:
        print("\n处理失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()