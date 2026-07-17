import os
import random
from PIL import Image, ImageDraw
import colorsys
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_output_directories():
    """创建输出目录"""
    output_dirs = ["./real_data", "./sim_data"]
    for d in output_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    return output_dirs

def generate_left_to_right_grid(width=1920, height=1080):
    """
    生成从左到右网格逐渐变密的网格图像
    """
    # 创建白色背景图像
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 水平方向网格线（从左到右逐渐变密）
    horizontal_lines = [0]
    x = 0
    grid_size = 150  # 起始网格大小
    min_size = 30    # 最小网格大小
    
    while x < width:
        # 逐渐减小网格大小
        grid_size = max(min_size, grid_size * 0.95)
        x += int(grid_size)
        horizontal_lines.append(x)
    horizontal_lines[-1] = width  # 确保最后一条线在边界
    
    # 垂直方向网格线（从上到下均匀分布）
    vertical_lines = []
    vertical_count = 8  # 垂直方向网格数量
    vertical_spacing = height // vertical_count
    
    for i in range(vertical_count + 1):
        vertical_lines.append(i * vertical_spacing)
    vertical_lines[-1] = height  # 确保最后一条线在边界
    
    # 生成每个网格的颜色
    colors = []
    for i in range(len(horizontal_lines) - 1):
        row_colors = []
        for j in range(len(vertical_lines) - 1):
            # 使用HSL色彩空间，相邻网格使用对比色
            base_hue = (i * 0.2 + j * 0.1) % 1.0
            
            # 奇偶行使用不同的亮度和饱和度
            if (i + j) % 2 == 0:
                hue = base_hue
                saturation = 0.7
                lightness = 0.5
            else:
                hue = (base_hue + 0.5) % 1.0  # 互补色
                saturation = 0.8
                lightness = 0.6
            
            # 转换为RGB
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            row_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        colors.append(row_colors)
    
    # 第一步：绘制所有彩色方块
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x1 = horizontal_lines[i]
            y1 = vertical_lines[j]
            x2 = horizontal_lines[i + 1]
            y2 = vertical_lines[j + 1]
            
            # 绘制彩色方块
            draw.rectangle([x1, y1, x2, y2], fill=colors[i][j])
    
    # 第二步：绘制5像素宽的白色网格线
    line_width = 5
    
    # 绘制垂直网格线
    for x in horizontal_lines:
        if 0 < x < width:  # 跳过边界
            x_left = x - line_width // 2
            x_right = x_left + line_width
            draw.rectangle([x_left, 0, x_right, height], fill='white')
    
    # 绘制水平网格线
    for y in vertical_lines:
        if 0 < y < height:  # 跳过边界
            y_top = y - line_width // 2
            y_bottom = y_top + line_width
            draw.rectangle([0, y_top, width, y_bottom], fill='white')
    
    # 第三步：绘制5像素宽的白色外边框
    draw.rectangle([0, 0, width-1, height-1], outline='white', width=line_width)
    
    return image, horizontal_lines, vertical_lines, colors

def generate_right_to_left_grid(width=1920, height=1080):
    """
    生成从右到左网格逐渐变密的网格图像
    """
    # 创建白色背景图像
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 水平方向网格线（从右到左逐渐变密，即从左到右逐渐变疏）
    horizontal_lines = [0]
    x = 0
    grid_size = 30  # 起始网格大小
    max_size = 150  # 最大网格大小
    
    while x < width:
        # 逐渐增大网格大小
        grid_size = min(max_size, grid_size * 1.05)
        x += int(grid_size)
        horizontal_lines.append(x)
    horizontal_lines[-1] = width
    
    # 垂直方向网格线（从上到下不均匀分布，增加变化）
    vertical_lines = [0]
    y = 0
    vertical_grid_sizes = [80, 120, 100, 90, 110, 95, 105, 85]  # 不同的网格大小
    
    for size in vertical_grid_sizes:
        y += size
        if y >= height:
            vertical_lines.append(height)
            break
        vertical_lines.append(y)
    
    if vertical_lines[-1] != height:
        vertical_lines[-1] = height
    
    # 生成每个网格的颜色
    colors = []
    for i in range(len(horizontal_lines) - 1):
        row_colors = []
        for j in range(len(vertical_lines) - 1):
            # 使用HSL色彩空间
            base_hue = (i * 0.15 + j * 0.2) % 1.0
            
            # 使用三色对比方案
            color_type = (i + j * 2) % 3
            
            if color_type == 0:  # 暖色调
                hue = (base_hue + 0.0) % 1.0
                saturation = 0.8
                lightness = 0.5
            elif color_type == 1:  # 冷色调
                hue = (base_hue + 0.33) % 1.0
                saturation = 0.7
                lightness = 0.6
            else:  # 中性色调
                hue = (base_hue + 0.66) % 1.0
                saturation = 0.6
                lightness = 0.55
            
            # 转换为RGB
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            row_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        colors.append(row_colors)
    
    # 第一步：绘制所有彩色方块
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x1 = horizontal_lines[i]
            y1 = vertical_lines[j]
            x2 = horizontal_lines[i + 1]
            y2 = vertical_lines[j + 1]
            
            # 绘制彩色方块
            draw.rectangle([x1, y1, x2, y2], fill=colors[i][j])
    
    # 第二步：绘制5像素宽的白色网格线
    line_width = 5
    
    # 绘制垂直网格线
    for x in horizontal_lines:
        if 0 < x < width:  # 跳过边界
            x_left = x - line_width // 2
            x_right = x_left + line_width
            draw.rectangle([x_left, 0, x_right, height], fill='white')
    
    # 绘制水平网格线
    for y in vertical_lines:
        if 0 < y < height:  # 跳过边界
            y_top = y - line_width // 2
            y_bottom = y_top + line_width
            draw.rectangle([0, y_top, width, y_bottom], fill='white')
    
    # 第三步：绘制5像素宽的白色外边框
    draw.rectangle([0, 0, width-1, height-1], outline='white', width=line_width)
    
    return image, horizontal_lines, vertical_lines, colors

def generate_uniform_grid(width=1920, height=1080, grid_size=100, color_scheme=1):
    """
    生成均匀网格图像，颜色方案与非均匀网格保持一致
    
    参数:
    width: 图像宽度
    height: 图像高度
    grid_size: 网格大小
    color_scheme: 颜色方案 (1: 与从左到右网格一致, 2: 与从右到左网格一致)
    """
    # 创建白色背景图像
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 计算网格线位置
    # 水平方向
    horizontal_lines = []
    x = 0
    while x < width:
        horizontal_lines.append(x)
        x += grid_size
    if horizontal_lines[-1] != width:
        horizontal_lines.append(width)
    
    # 垂直方向
    vertical_lines = []
    y = 0
    while y < height:
        vertical_lines.append(y)
        y += grid_size
    if vertical_lines[-1] != height:
        vertical_lines.append(height)
    
    # 生成每个网格的颜色，保持与非均匀网格一致的颜色分布
    colors = []
    for i in range(len(horizontal_lines) - 1):
        row_colors = []
        for j in range(len(vertical_lines) - 1):
            if color_scheme == 1:
                # 方案1: 与从左到右网格相同的颜色方案
                base_hue = (i * 0.2 + j * 0.1) % 1.0
                
                if (i + j) % 2 == 0:
                    hue = base_hue
                    saturation = 0.7
                    lightness = 0.5
                else:
                    hue = (base_hue + 0.5) % 1.0
                    saturation = 0.8
                    lightness = 0.6
            else:
                # 方案2: 与从右到左网格相同的颜色方案
                base_hue = (i * 0.15 + j * 0.2) % 1.0
                color_type = (i + j * 2) % 3
                
                if color_type == 0:  # 暖色调
                    hue = (base_hue + 0.0) % 1.0
                    saturation = 0.8
                    lightness = 0.5
                elif color_type == 1:  # 冷色调
                    hue = (base_hue + 0.33) % 1.0
                    saturation = 0.7
                    lightness = 0.6
                else:  # 中性色调
                    hue = (base_hue + 0.66) % 1.0
                    saturation = 0.6
                    lightness = 0.55
            
            # 转换为RGB
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            row_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        colors.append(row_colors)
    
    # 第一步：绘制所有彩色方块
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            x1 = horizontal_lines[i]
            y1 = vertical_lines[j]
            x2 = horizontal_lines[i + 1]
            y2 = vertical_lines[j + 1]
            
            # 绘制彩色方块
            draw.rectangle([x1, y1, x2, y2], fill=colors[i][j])
    
    # 第二步：绘制5像素宽的白色网格线
    line_width = 5
    
    # 绘制垂直网格线
    for x in horizontal_lines:
        if 0 < x < width:  # 跳过边界
            x_left = x - line_width // 2
            x_right = x_left + line_width
            draw.rectangle([x_left, 0, x_right, height], fill='white')
    
    # 绘制水平网格线
    for y in vertical_lines:
        if 0 < y < height:  # 跳过边界
            y_top = y - line_width // 2
            y_bottom = y_top + line_width
            draw.rectangle([0, y_top, width, y_bottom], fill='white')
    
    # 第三步：绘制5像素宽的白色外边框
    draw.rectangle([0, 0, width-1, height-1], outline='white', width=line_width)
    
    return image, horizontal_lines, vertical_lines, colors

def main():
    """主函数，生成并保存所有网格图像"""
    
    # 创建输出目录
    output_dirs = create_output_directories()
    
    # 生成第一种网格：从左到右逐渐变密
    print("正在生成从左到右逐渐变密的网格图像...")
    image1, h_lines1, v_lines1, colors1 = generate_left_to_right_grid(1920, 1080)
    for d in output_dirs:
        output_path1 = os.path.join(d, "1_pro.png")
        image1.save(output_path1)
        print(f"已保存到: {output_path1}")
    print(f"水平网格数: {len(h_lines1)-1}, 垂直网格数: {len(v_lines1)-1}")
    
    # 生成第二种网格：从右到左逐渐变密
    print("\n正在生成从右到左逐渐变密的网格图像...")
    image2, h_lines2, v_lines2, colors2 = generate_right_to_left_grid(1920, 1080)
    for d in output_dirs:
        output_path2 = os.path.join(d, "2_pro.png")
        image2.save(output_path2)
        print(f"已保存到: {output_path2}")
    print(f"水平网格数: {len(h_lines2)-1}, 垂直网格数: {len(v_lines2)-1}")
    
    # 生成第三种网格：均匀网格，使用与第一种网格相同的颜色方案
    print("\n正在生成均匀网格图像(100px，颜色方案1)...")
    image3, h_lines3, v_lines3, colors3 = generate_uniform_grid(1920, 1080, grid_size=100, color_scheme=1)
    for d in output_dirs:
        output_path3 = os.path.join(d, "3_pro.png")
        image3.save(output_path3)
        print(f"已保存到: {output_path3}")
    print(f"水平网格数: {len(h_lines3)-1}, 垂直网格数: {len(v_lines3)-1}")
    
    # 生成第四张图像：全红色的图像 (仅保存在 real_data 中)
    print("\n正在生成全红色图像...")
    image4 = Image.new('RGB', (1920, 1080), (255, 0, 0))
    output_path4 = os.path.join("./real_data", "4_proj.png")
    image4.save(output_path4)
    print(f"已保存到: {output_path4}")

    # # 生成第四种网格：小尺寸均匀网格，使用与第二种网格相同的颜色方案
    # print("\n正在生成小尺寸均匀网格图像(50px，颜色方案2)...")
    # image4, h_lines4, v_lines4, colors4 = generate_uniform_grid(1920, 1080, grid_size=50, color_scheme=2)
    # output_path4 = os.path.join(output_dir, "4_pro.png")
    # image4.save(output_path4)
    # print(f"已保存到: {output_path4}")
    # print(f"水平网格数: {len(h_lines4)-1}, 垂直网格数: {len(v_lines4)-1}")
    
    # # 生成第五种网格：大尺寸均匀网格，使用与第一种网格相同的颜色方案
    # print("\n正在生成大尺寸均匀网格图像(150px，颜色方案1)...")
    # image5, h_lines5, v_lines5, colors5 = generate_uniform_grid(1920, 1080, grid_size=150, color_scheme=1)
    # output_path5 = os.path.join(output_dir, "5_pro.png")
    # image5.save(output_path5)
    # print(f"已保存到: {output_path5}")
    # print(f"水平网格数: {len(h_lines5)-1}, 垂直网格数: {len(v_lines5)-1}")
    
    # 打印网格统计信息
    print("\n" + "="*60)
    print("网格统计信息:")
    print("="*60)
    
    print(f"\n1_pro.png (从左到右逐渐变密的网格):")
    print(f"  水平方向: 起始网格宽度 = {h_lines1[1]-h_lines1[0]}px")
    print(f"  水平方向: 结束网格宽度 = {h_lines1[-1]-h_lines1[-2]}px")
    print(f"  垂直方向: 平均网格高度 = {1080//(len(v_lines1)-1)}px")
    
    print(f"\n2_pro.png (从右到左逐渐变密的网格):")
    print(f"  水平方向: 起始网格宽度 = {h_lines2[1]-h_lines2[0]}px")
    print(f"  水平方向: 结束网格宽度 = {h_lines2[-1]-h_lines2[-2]}px")
    
    print(f"\n3_pro.png (均匀网格100px，颜色方案1):")
    print(f"  水平方向: 网格宽度 = {h_lines3[1]-h_lines3[0]}px")
    print(f"  垂直方向: 网格高度 = {v_lines3[1]-v_lines3[0]}px")
    print(f"  网格总数: {(len(h_lines3)-1) * (len(v_lines3)-1)}")
    
    # print(f"\n4_pro.png (均匀网格50px，颜色方案2):")
    # print(f"  水平方向: 网格宽度 = {h_lines4[1]-h_lines4[0]}px")
    # print(f"  垂直方向: 网格高度 = {v_lines4[1]-v_lines4[0]}px")
    # print(f"  网格总数: {(len(h_lines4)-1) * (len(v_lines4)-1)}")
    
    # print(f"\n5_pro.png (均匀网格150px，颜色方案1):")
    # print(f"  水平方向: 网格宽度 = {h_lines5[1]-h_lines5[0]}px")
    # print(f"  垂直方向: 网格高度 = {v_lines5[1]-v_lines5[0]}px")
    # print(f"  网格总数: {(len(h_lines5)-1) * (len(v_lines5)-1)}")
    
    print("\n" + "="*60)
    print(f"所有图像已保存到以下目录:")
    for d in output_dirs:
        print(f"  - {os.path.abspath(d)}")
    print("\n颜色方案说明:")
    print("1_pro.png, 3_pro.png, 5_pro.png: 使用相同的互补色方案")
    print("2_pro.png, 4_pro.png: 使用相同的三色对比方案")
    print("="*60)
    
    # 验证颜色方案一致性
    print("\n颜色分布验证:")
    print("-" * 40)
    
    # 统计1_pro.png和3_pro.png的颜色特征
    def get_color_stats(colors):
        """获取颜色统计信息"""
        r_values, g_values, b_values = [], [], []
        for row in colors:
            for r, g, b in row:
                r_values.append(r)
                g_values.append(g)
                b_values.append(b)
        
        return {
            'avg_r': sum(r_values) / len(r_values),
            'avg_g': sum(g_values) / len(g_values),
            'avg_b': sum(b_values) / len(b_values),
            'min_r': min(r_values),
            'max_r': max(r_values),
            'num_colors': len(r_values)
        }
    
    stats1 = get_color_stats(colors1)
    stats3 = get_color_stats(colors3)
    # stats5 = get_color_stats(colors5)
    
    print(f"1_pro.png: 平均颜色(RGB) = ({stats1['avg_r']:.1f}, {stats1['avg_g']:.1f}, {stats1['avg_b']:.1f})")
    print(f"3_pro.png: 平均颜色(RGB) = ({stats3['avg_r']:.1f}, {stats3['avg_g']:.1f}, {stats3['avg_b']:.1f})")
    # print(f"5_pro.png: 平均颜色(RGB) = ({stats5['avg_r']:.1f}, {stats5['avg_g']:.1f}, {stats5['avg_b']:.1f})")
    
    print("\n颜色方案一致性检查:")
    print(f"1_pro.png 和 3_pro.png 平均颜色差异: "
          f"R:{abs(stats1['avg_r']-stats3['avg_r']):.1f}, "
          f"G:{abs(stats1['avg_g']-stats3['avg_g']):.1f}, "
          f"B:{abs(stats1['avg_b']-stats3['avg_b']):.1f}")
    # print(f"1_pro.png 和 5_pro.png 平均颜色差异: "
    #       f"R:{abs(stats1['avg_r']-stats5['avg_r']):.1f}, "
    #       f"G:{abs(stats1['avg_g']-stats5['avg_g']):.1f}, "
    #       f"B:{abs(stats1['avg_b']-stats5['avg_b']):.1f}")

    # === 绘制并保存到 ./data ===
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("从左到右逐渐变密")
    plt.imshow(image1)

    plt.subplot(2, 2, 2)
    plt.title("从右到左逐渐变密")
    plt.imshow(image2)

    plt.subplot(2, 2, 3)
    plt.title("均匀网格(100px)")
    plt.imshow(image3)

    plt.subplot(2, 2, 4)
    plt.title("全红色图像")
    plt.imshow(image4)

    plt.tight_layout()
    if not os.path.exists("./data"):
        os.makedirs("./data")
    plt.savefig("./data/generated_meshes.png")
    # plt.show()

if __name__ == "__main__":
    main()