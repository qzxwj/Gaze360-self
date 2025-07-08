import os
import numpy as np
import cv2
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import model
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import random
from collections import defaultdict
import traceback
import warnings

# 忽略一些常见的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def gazeto3d(gaze):
    """将视线角度转换为3D坐标"""
    try:
        # 确保输入是numpy数组
        if torch.is_tensor(gaze):
            gaze = gaze.cpu().detach().numpy()

        gaze = np.array(gaze, dtype=float).flatten()

        if len(gaze) < 2:
            print(f"Invalid gaze data length: {len(gaze)}")
            return None

        gaze_gt = np.zeros([3])
        gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
        gaze_gt[1] = -np.sin(gaze[1])
        gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
        return gaze_gt
    except Exception as e:
        print(f"Error in gazeto3d: {e}")
        return None


def angular(gaze, label):
    """计算预测的视线与真实视线之间的角度误差"""
    try:
        # 确保都是numpy数组
        if torch.is_tensor(gaze):
            gaze = gaze.cpu().detach().numpy()
        if torch.is_tensor(label):
            label = label.cpu().detach().numpy()

        gaze = np.array(gaze, dtype=float).flatten()
        label = np.array(label, dtype=float).flatten()

        if len(gaze) == 0 or len(label) == 0:
            return None

        total = np.sum(gaze * label)
        norm_gaze = np.linalg.norm(gaze)
        norm_label = np.linalg.norm(label)

        if norm_gaze == 0 or norm_label == 0:
            return 0.0

        cos_angle = np.clip(total / (norm_gaze * norm_label), -0.9999999, 0.9999999)
        return np.arccos(np.abs(cos_angle)) * 180 / np.pi
    except Exception as e:
        print(f"Error in angular calculation: {e}")
        return None


def ensure_dir_exists(path):
    """确保目录存在，如果不存在则创建"""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False


def safe_tensor_to_numpy(tensor):
    """安全地将tensor转换为numpy数组"""
    try:
        if torch.is_tensor(tensor):
            return tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor.copy()
        else:
            return np.array(tensor)
    except Exception as e:
        print(f"Error converting tensor to numpy: {e}")
        return None


def create_gaze_heatmap(image_shape, gaze_3d, intensity=100):
    """创建视线注意力热力图"""
    try:
        if gaze_3d is None or len(image_shape) < 2:
            return np.zeros(image_shape[:2], dtype=np.uint8), (0, 0)

        h, w = image_shape[:2]
        center_x, center_y = w // 2, h // 2

        # 计算视线在图像平面上的投影点
        if abs(gaze_3d[2]) > 1e-6:
            proj_x = center_x + (gaze_3d[0] / abs(gaze_3d[2])) * min(w, h) * 0.3
            proj_y = center_y - (gaze_3d[1] / abs(gaze_3d[2])) * min(w, h) * 0.3
        else:
            proj_x, proj_y = center_x, center_y

        # 限制投影点在图像范围内
        proj_x = max(0, min(w - 1, proj_x))
        proj_y = max(0, min(h - 1, proj_y))

        # 创建高斯热力图
        y, x = np.ogrid[:h, :w]
        sigma = min(w, h) * 0.1
        heatmap = np.exp(-((x - proj_x) ** 2 + (y - proj_y) ** 2) / (2 * sigma ** 2))
        heatmap = (heatmap * intensity).astype(np.uint8)

        return heatmap, (int(proj_x), int(proj_y))
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return np.zeros(image_shape[:2], dtype=np.uint8), (0, 0)


def process_image_for_visualization(image):
    """处理图像用于可视化"""
    try:
        # 转换tensor到numpy
        if torch.is_tensor(image):
            image = image.cpu().detach().numpy()

        if image is None:
            return None

        # 复制数组避免修改原始数据
        image = image.copy()

        # 处理维度
        if len(image.shape) == 4:  # NCHW
            image = image[0]  # 取第一个样本
        if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW
            if image.shape[0] == 1:  # 灰度图
                image = image[0]
            else:  # RGB图
                image = np.transpose(image, (1, 2, 0))

        # 确保像素值在正确范围
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # 确保是3通道图像
        if len(image.shape) == 2:  # 灰度图转RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 验证图像有效性
        if image.shape[0] == 0 or image.shape[1] == 0:
            return None

        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def safe_save_plot(fig, save_path, filename, dpi=300):
    """安全保存matplotlib图像"""
    try:
        if not ensure_dir_exists(save_path):
            return False

        full_path = os.path.join(save_path, filename)
        plt.figure(fig.number)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {full_path}")
        return True
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
        try:
            plt.close(fig)
        except:
            pass
        return False


def visualize_enhanced_3d_gaze(gaze_pred, gaze_gt, name, save_dir, angle_error):
    """增强的3D视线可视化 - 使用简单的quiver替代自定义箭头"""
    try:
        gaze_pred_3d = gazeto3d(gaze_pred)
        gaze_gt_3d = gazeto3d(gaze_gt)

        if gaze_pred_3d is None or gaze_gt_3d is None:
            print(f"Failed to convert gaze to 3D for {name}")
            return False

        fig = plt.figure(figsize=(15, 10))

        # 主3D视图 - 使用简单的quiver3D
        ax1 = fig.add_subplot(221, projection='3d')

        try:
            # 绘制坐标系
            ax1.quiver(0, 0, 0, 0.8, 0, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1, linewidth=1)
            ax1.quiver(0, 0, 0, 0, 0.8, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1, linewidth=1)
            ax1.quiver(0, 0, 0, 0, 0, 0.8, color='gray', alpha=0.3, arrow_length_ratio=0.1, linewidth=1)

            # 绘制预测视线（红色箭头）- 使用quiver3D
            ax1.quiver(0, 0, 0, gaze_pred_3d[0], gaze_pred_3d[1], gaze_pred_3d[2],
                       color='red', arrow_length_ratio=0.1, linewidth=3, label='Predicted')

            # 绘制真实视线（蓝色箭头）
            ax1.quiver(0, 0, 0, gaze_gt_3d[0], gaze_gt_3d[1], gaze_gt_3d[2],
                       color='blue', arrow_length_ratio=0.1, linewidth=3, label='Ground Truth')

            # 绘制简化的球面网格
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_sphere = np.outer(np.cos(u), np.sin(v)) * 0.9
            y_sphere = np.outer(np.sin(u), np.sin(v)) * 0.9
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.9
            ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightgray', linewidth=0.5)

            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])
            ax1.set_zlim([-1, 1])
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'3D Gaze Vectors\nError: {angle_error:.2f}°')

            # 添加文本标签
            ax1.text(gaze_pred_3d[0], gaze_pred_3d[1], gaze_pred_3d[2], 'Pred', color='red')
            ax1.text(gaze_gt_3d[0], gaze_gt_3d[1], gaze_gt_3d[2], 'GT', color='blue')

        except Exception as e:
            print(f"Error in 3D subplot: {e}")

        # XY平面投影
        ax2 = fig.add_subplot(222)
        try:
            ax2.arrow(0, 0, gaze_pred_3d[0], gaze_pred_3d[1], head_width=0.05, head_length=0.05,
                      fc='red', ec='red', label='Predicted', linewidth=2)
            ax2.arrow(0, 0, gaze_gt_3d[0], gaze_gt_3d[1], head_width=0.05, head_length=0.05,
                      fc='blue', ec='blue', label='Ground Truth', linewidth=2)
            ax2.set_xlim([-1, 1])
            ax2.set_ylim([-1, 1])
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('XY Plane Projection')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_aspect('equal')

            # 添加单位圆
            circle = plt.Circle((0, 0), 1, color='gray', fill=False, alpha=0.3)
            ax2.add_patch(circle)

        except Exception as e:
            print(f"Error in XY subplot: {e}")

        # XZ平面投影
        ax3 = fig.add_subplot(223)
        try:
            ax3.arrow(0, 0, gaze_pred_3d[0], gaze_pred_3d[2], head_width=0.05, head_length=0.05,
                      fc='red', ec='red', label='Predicted', linewidth=2)
            ax3.arrow(0, 0, gaze_gt_3d[0], gaze_gt_3d[2], head_width=0.05, head_length=0.05,
                      fc='blue', ec='blue', label='Ground Truth', linewidth=2)
            ax3.set_xlim([-1, 1])
            ax3.set_ylim([-1, 1])
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.set_title('XZ Plane Projection')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_aspect('equal')

            # 添加单位圆
            circle = plt.Circle((0, 0), 1, color='gray', fill=False, alpha=0.3)
            ax3.add_patch(circle)

        except Exception as e:
            print(f"Error in XZ subplot: {e}")

        # YZ平面投影
        ax4 = fig.add_subplot(224)
        try:
            ax4.arrow(0, 0, gaze_pred_3d[1], gaze_pred_3d[2], head_width=0.05, head_length=0.05,
                      fc='red', ec='red', label='Predicted', linewidth=2)
            ax4.arrow(0, 0, gaze_gt_3d[1], gaze_gt_3d[2], head_width=0.05, head_length=0.05,
                      fc='blue', ec='blue', label='Ground Truth', linewidth=2)
            ax4.set_xlim([-1, 1])
            ax4.set_ylim([-1, 1])
            ax4.set_xlabel('Y')
            ax4.set_ylabel('Z')
            ax4.set_title('YZ Plane Projection')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_aspect('equal')

            # 添加单位圆
            circle = plt.Circle((0, 0), 1, color='gray', fill=False, alpha=0.3)
            ax4.add_patch(circle)

        except Exception as e:
            print(f"Error in YZ subplot: {e}")

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, "3d_visualization")
        filename = f"{os.path.basename(name)}_3d_enhanced.png"
        return safe_save_plot(fig, save_path, filename)

    except Exception as e:
        print(f"Error in enhanced 3D visualization for {name}: {e}")
        traceback.print_exc()
        return False


def visualize_gaze_on_image_enhanced(image, gaze_pred, gaze_gt, name, save_dir):
    """增强的图像上视线可视化"""
    try:
        # 处理图像
        processed_image = process_image_for_visualization(image)
        if processed_image is None:
            print(f"Failed to process image for {name}")
            return False

        # 转换为BGR用于OpenCV
        image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]

        # 获取3D方向
        gaze_pred_3d = gazeto3d(gaze_pred)
        gaze_gt_3d = gazeto3d(gaze_gt)

        if gaze_pred_3d is None or gaze_gt_3d is None:
            print(f"Failed to convert gaze to 3D for {name}")
            return False

        # 创建多个可视化版本
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        try:
            # 1. 原始图像 + 2D箭头
            ax1 = axes[0, 0]
            image_arrows = image_bgr.copy()

            # 绘制箭头
            center = (w // 2, h // 2)
            scale = min(w, h) * 0.3

            # 预测箭头（红色）
            end_pred = (int(center[0] + gaze_pred_3d[0] * scale),
                        int(center[1] - gaze_pred_3d[1] * scale))
            cv2.arrowedLine(image_arrows, center, end_pred, (0, 0, 255), 3, tipLength=0.1)

            # 真实箭头（蓝色）
            end_gt = (int(center[0] + gaze_gt_3d[0] * scale),
                      int(center[1] - gaze_gt_3d[1] * scale))
            cv2.arrowedLine(image_arrows, center, end_gt, (255, 0, 0), 3, tipLength=0.1)

            # 添加圆心和标签
            cv2.circle(image_arrows, center, 5, (0, 255, 0), -1)
            cv2.putText(image_arrows, 'Red: Pred', (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image_arrows, 'Blue: GT', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            ax1.imshow(cv2.cvtColor(image_arrows, cv2.COLOR_BGR2RGB))
            ax1.set_title('Gaze Arrows on Image')
            ax1.axis('off')
        except Exception as e:
            print(f"Error in arrow visualization: {e}")

        try:
            # 2. 预测热力图
            ax2 = axes[0, 1]
            heatmap_pred, proj_pred = create_gaze_heatmap(processed_image.shape, gaze_pred_3d)
            heatmap_colored = cv2.applyColorMap(heatmap_pred, cv2.COLORMAP_JET)
            overlay_pred = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)

            ax2.imshow(cv2.cvtColor(overlay_pred, cv2.COLOR_BGR2RGB))
            ax2.set_title('Predicted Gaze Heatmap')
            ax2.axis('off')
        except Exception as e:
            print(f"Error in predicted heatmap: {e}")

        try:
            # 3. 真实热力图
            ax3 = axes[0, 2]
            heatmap_gt, proj_gt = create_gaze_heatmap(processed_image.shape, gaze_gt_3d)
            heatmap_colored_gt = cv2.applyColorMap(heatmap_gt, cv2.COLORMAP_JET)
            overlay_gt = cv2.addWeighted(image_bgr, 0.6, heatmap_colored_gt, 0.4, 0)

            ax3.imshow(cv2.cvtColor(overlay_gt, cv2.COLOR_BGR2RGB))
            ax3.set_title('Ground Truth Gaze Heatmap')
            ax3.axis('off')
        except Exception as e:
            print(f"Error in ground truth heatmap: {e}")

        try:
            # 4. 误差可视化
            ax4 = axes[1, 0]
            error_image = image_bgr.copy()

            # 绘制误差向量
            cv2.arrowedLine(error_image, end_gt, end_pred, (0, 255, 255), 2, tipLength=0.1)
            cv2.circle(error_image, end_gt, 8, (255, 0, 0), 2)
            cv2.circle(error_image, end_pred, 8, (0, 0, 255), 2)

            # 添加误差信息
            angle_error = angular(gaze_pred_3d, gaze_gt_3d)
            if angle_error is not None:
                cv2.putText(error_image, f'Error: {angle_error:.2f}°', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(error_image, 'Yellow: Error Vector', (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            ax4.imshow(cv2.cvtColor(error_image, cv2.COLOR_BGR2RGB))
            ax4.set_title(f'Gaze Error Visualization')
            ax4.axis('off')
        except Exception as e:
            print(f"Error in error visualization: {e}")

        try:
            # 5. 方向向量对比
            ax5 = axes[1, 1]
            directions = ['X', 'Y', 'Z']
            pred_values = gaze_pred_3d
            gt_values = gaze_gt_3d

            x = np.arange(len(directions))
            width = 0.35

            bars1 = ax5.bar(x - width / 2, pred_values, width, label='Predicted', color='red', alpha=0.7)
            bars2 = ax5.bar(x + width / 2, gt_values, width, label='Ground Truth', color='blue', alpha=0.7)

            # 添加数值标签
            for bar, val in zip(bars1, pred_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            for bar, val in zip(bars2, gt_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

            ax5.set_xlabel('Axis')
            ax5.set_ylabel('Value')
            ax5.set_title('3D Direction Components')
            ax5.set_xticks(x)
            ax5.set_xticklabels(directions)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error in direction comparison: {e}")

        try:
            # 6. 角度对比和统计
            ax6 = axes[1, 2]

            # 计算球坐标角度
            def cart_to_spherical(xyz):
                try:
                    x, y, z = xyz
                    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    theta = np.arccos(np.clip(z / r, -1, 1)) if r > 0 else 0  # polar angle
                    phi = np.arctan2(y, x)  # azimuthal angle
                    return theta * 180 / np.pi, phi * 180 / np.pi
                except:
                    return 0, 0

            theta_pred, phi_pred = cart_to_spherical(gaze_pred_3d)
            theta_gt, phi_gt = cart_to_spherical(gaze_gt_3d)

            # 创建统计表格
            angle_error = angular(gaze_pred_3d,
                                  gaze_gt_3d) if gaze_pred_3d is not None and gaze_gt_3d is not None else 0

            stats_text = f"""Gaze Statistics:
Angular Error: {angle_error:.3f}°

Predicted:
  Polar (θ): {theta_pred:.2f}°
  Azimuth (φ): {phi_pred:.2f}°

Ground Truth:
  Polar (θ): {theta_gt:.2f}°
  Azimuth (φ): {phi_gt:.2f}°

3D Vector Norm:
  Pred: {np.linalg.norm(gaze_pred_3d):.3f}
  GT: {np.linalg.norm(gaze_gt_3d):.3f}

Dot Product: {np.dot(gaze_pred_3d, gaze_gt_3d):.3f}"""

            ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax6.set_title('Detailed Statistics')
            ax6.axis('off')

        except Exception as e:
            print(f"Error in statistics display: {e}")

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, "image_visualization")
        filename = f"{os.path.basename(name)}_image_enhanced.png"
        return safe_save_plot(fig, save_path, filename)

    except Exception as e:
        print(f"Error in enhanced image visualization for {name}: {e}")
        traceback.print_exc()
        return False


def create_error_distribution_plot(errors, names, save_dir):
    """创建误差分布图"""
    try:
        if not errors:
            print("No errors to plot")
            return False

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        try:
            # 误差直方图
            ax1 = axes[0, 0]
            n_bins = min(30, max(10, len(errors) // 3))
            counts, bins, patches = ax1.hist(errors, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Angular Error (degrees)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Error Distribution')
            ax1.grid(True, alpha=0.3)

            # 误差统计信息
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}°')
            ax1.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}°')
            ax1.legend()
        except Exception as e:
            print(f"Error in histogram: {e}")

        try:
            # 累积分布
            ax2 = axes[0, 1]
            sorted_errors = np.sort(errors)
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(sorted_errors, cumulative, 'b-', linewidth=2)
            ax2.set_xlabel('Angular Error (degrees)')
            ax2.set_ylabel('Cumulative Probability')
            ax2.set_title('Cumulative Error Distribution')
            ax2.grid(True, alpha=0.3)

            # 添加百分位点
            p50 = np.percentile(errors, 50)
            p90 = np.percentile(errors, 90)
            p95 = np.percentile(errors, 95)
            ax2.axvline(p50, color='green', linestyle='--', linewidth=2, label=f'50th: {p50:.2f}°')
            ax2.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'90th: {p90:.2f}°')
            ax2.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'95th: {p95:.2f}°')
            ax2.legend()
        except Exception as e:
            print(f"Error in cumulative plot: {e}")

        try:
            # 误差随样本变化
            ax3 = axes[1, 0]
            sample_indices = range(len(errors))
            colors = ['red' if e > np.percentile(errors, 90) else 'orange' if e > np.percentile(errors, 75) else 'green'
                      for e in errors]
            ax3.scatter(sample_indices, errors, alpha=0.6, s=30, c=colors)
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Angular Error (degrees)')
            ax3.set_title('Error per Sample (Red: >90th, Orange: >75th, Green: ≤75th)')
            ax3.grid(True, alpha=0.3)

            # 添加趋势线
            z = np.polyfit(sample_indices, errors, 1)
            p = np.poly1d(z)
            ax3.plot(sample_indices, p(sample_indices), "r--", alpha=0.8, label=f'Trend: {z[0]:.6f}x + {z[1]:.3f}')
            ax3.legend()
        except Exception as e:
            print(f"Error in scatter plot: {e}")

        try:
            # 箱线图和小提琴图结合
            ax4 = axes[1, 1]

            # 创建小提琴图
            parts = ax4.violinplot([errors], positions=[1], widths=0.6, showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)

            # 叠加箱线图
            box_plot = ax4.boxplot([errors], positions=[1], widths=0.3, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightcoral')
            box_plot['boxes'][0].set_alpha(0.7)

            ax4.set_xticks([1])
            ax4.set_xticklabels(['Angular Error'])
            ax4.set_ylabel('Error (degrees)')
            ax4.set_title('Error Distribution (Violin + Box Plot)')
            ax4.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error in violin/box plot: {e}")

        # 添加统计信息文本
        try:
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            p50 = np.percentile(errors, 50)
            p90 = np.percentile(errors, 90)
            p95 = np.percentile(errors, 95)
            p99 = np.percentile(errors, 99)

            stats_text = f"""Statistics Summary:
Samples: {len(errors)}
Mean: {mean_error:.3f}°
Median: {median_error:.3f}°
Std: {std_error:.3f}°
Min: {np.min(errors):.3f}°
Max: {np.max(errors):.3f}°

Percentiles:
50th: {p50:.3f}°
90th: {p90:.3f}°
95th: {p95:.3f}°
99th: {p99:.3f}°

Quality Metrics:
<5°: {np.sum(np.array(errors) < 5) / len(errors) * 100:.1f}%
<10°: {np.sum(np.array(errors) < 10) / len(errors) * 100:.1f}%
<15°: {np.sum(np.array(errors) < 15) / len(errors) * 100:.1f}%"""

            fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        except Exception as e:
            print(f"Error adding statistics text: {e}")

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(save_dir, "statistics")
        filename = "error_distribution.png"
        return safe_save_plot(fig, save_path, filename)

    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        traceback.print_exc()
        return False


def create_comparison_grid(selected_data, save_dir):
    """创建对比网格图"""
    try:
        if not selected_data:
            print("No data for comparison grid")
            return False

        n_samples = len(selected_data)
        cols = min(5, n_samples)
        rows = (n_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        # 处理单行或单列的情况
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i in range(n_samples):
            try:
                image, gaze_pred, gaze_gt, name, error = selected_data[i]
                row = i // cols
                col = i % cols
                ax = axes[row][col]

                # 处理图像
                processed_image = process_image_for_visualization(image)
                if processed_image is None:
                    ax.text(0.5, 0.5, 'Image Error', ha='center', va='center')
                    ax.set_title(f'{os.path.basename(name)}\nError: {error:.2f}°', fontsize=10)
                    ax.axis('off')
                    continue

                # 转换为BGR用于OpenCV处理
                image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                h, w = image_bgr.shape[:2]

                # 绘制箭头
                gaze_pred_3d = gazeto3d(gaze_pred)
                gaze_gt_3d = gazeto3d(gaze_gt)

                if gaze_pred_3d is not None and gaze_gt_3d is not None:
                    center = (w // 2, h // 2)
                    scale = min(w, h) * 0.3

                    # 预测箭头（红色）
                    end_pred = (int(center[0] + gaze_pred_3d[0] * scale),
                                int(center[1] - gaze_pred_3d[1] * scale))
                    cv2.arrowedLine(image_bgr, center, end_pred, (0, 0, 255), 2, tipLength=0.1)

                    # 真实箭头（蓝色）
                    end_gt = (int(center[0] + gaze_gt_3d[0] * scale),
                              int(center[1] - gaze_gt_3d[1] * scale))
                    cv2.arrowedLine(image_bgr, center, end_gt, (255, 0, 0), 2, tipLength=0.1)

                    # 添加误差文本
                    cv2.putText(image_bgr, f'{error:.1f}°', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # 添加质量指示器
                    quality_color = (0, 255, 0) if error < 10 else (0, 165, 255) if error < 20 else (0, 0, 255)
                    cv2.circle(image_bgr, (w - 30, 30), 15, quality_color, -1)

                # 显示图像
                ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                quality_text = "Good" if error < 10 else "Medium" if error < 20 else "Poor"
                ax.set_title(f'{os.path.basename(name)}\nError: {error:.2f}° ({quality_text})', fontsize=10)
                ax.axis('off')

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        # 隐藏多余的子图
        for i in range(n_samples, rows * cols):
            try:
                row = i // cols
                col = i % cols
                ax = axes[row][col]
                ax.axis('off')
            except:
                pass

        # 添加图例
        legend_text = """Legend:
Red Arrow: Predicted Gaze
Blue Arrow: Ground Truth Gaze
Circle Color: Green (<10°), Orange (<20°), Red (≥20°)"""

        fig.text(0.02, 0.98, legend_text, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为图例留空间

        # 保存图像
        save_path = os.path.join(save_dir, "comparison")
        filename = "gaze_comparison_grid.png"
        return safe_save_plot(fig, save_path, filename)

    except Exception as e:
        print(f"Error creating comparison grid: {e}")
        traceback.print_exc()
        return False


def select_representative_samples(all_data, save_best_worst=5):
    """安全地选择代表性样本，避免numpy数组比较问题"""
    try:
        if not all_data:
            return []

        # 使用索引而不是直接比较数据来避免numpy数组比较问题
        indexed_data = [(i, data) for i, data in enumerate(all_data)]
        sorted_indexed = sorted(indexed_data, key=lambda x: x[1][4])  # 按误差排序

        n_samples = len(sorted_indexed)

        # 获取索引
        best_indices = set()
        worst_indices = set()
        mid_indices = set()

        # 最好的样本索引
        for i in range(min(save_best_worst, n_samples)):
            best_indices.add(sorted_indexed[i][0])

        # 最坏的样本索引
        for i in range(max(0, n_samples - save_best_worst), n_samples):
            worst_indices.add(sorted_indexed[i][0])

        # 中等样本索引
        if n_samples >= save_best_worst * 2:
            mid_start = n_samples // 2 - save_best_worst // 2
            for i in range(mid_start, min(mid_start + save_best_worst, n_samples)):
                mid_indices.add(sorted_indexed[i][0])

        # 随机样本索引
        all_selected_indices = best_indices | worst_indices | mid_indices
        remaining_indices = [i for i in range(n_samples) if i not in all_selected_indices]

        random_indices = set()
        if remaining_indices:
            sample_count = min(save_best_worst, len(remaining_indices))
            random_indices = set(random.sample(remaining_indices, sample_count))

        # 收集所有选中的样本
        final_indices = best_indices | worst_indices | mid_indices | random_indices
        selected_samples = [all_data[i] for i in final_indices]

        print(f"Selected {len(selected_samples)} representative samples")
        print(f"  Best: {len(best_indices)}, Worst: {len(worst_indices)}")
        print(f"  Mid: {len(mid_indices)}, Random: {len(random_indices)}")

        return selected_samples

    except Exception as e:
        print(f"Error selecting representative samples: {e}")
        # 如果选择失败，返回前几个样本
        return all_data[:min(20, len(all_data))] if all_data else []


def visualize(config_path, max_visualize=20, save_best_worst=5):
    """
    主可视化函数
    max_visualize: 最大可视化样本数
    save_best_worst: 保存最好和最坏样本的数量
    """
    try:
        # 读取配置文件
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        readername = config["reader"]
        dataloader = importlib.import_module("reader." + readername)

        config = config["test"]
        imagepath = config["data"]["image"]
        labelpath = config["data"]["label"]
        modelname = config["load"]["model_name"]

        loadpath = config["load"]["load_path"]
        savepath = "/root/autodl-tmp/Gaze360/visual_enhanced"

        if not ensure_dir_exists(savepath):
            print("Failed to create save directory")
            return

        # 加载数据
        print("Loading dataset...")
        dataset = dataloader.txtload(labelpath, imagepath, 32, num_workers=4, header=True)

        begin = config["load"]["begin_step"]
        end = config["load"]["end_step"]
        step = config["load"]["steps"]

        for saveiter in range(begin, end + step, step):
            try:
                print(f"Loading model for iteration {saveiter}")
                net = model.GazeLSTM()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                statedict = torch.load(
                    os.path.join(loadpath, f"checkpoint", f"Iter_{saveiter}_{modelname}.pt"),
                    map_location=device
                )
                net.to(device)
                net.load_state_dict(statedict)
                net.eval()

                # 创建迭代特定的保存目录
                iter_save_path = os.path.join(savepath, f"iter_{saveiter}")
                if not ensure_dir_exists(iter_save_path):
                    continue

                accs = 0
                count = 0
                all_errors = []
                all_data = []

                print(f"Starting visualization for iteration {saveiter}")

                with torch.no_grad():
                    for j, (data, label) in enumerate(dataset):
                        try:
                            img = data["face"].to(device)
                            names = data["name"]
                            gts = label.to(device)
                            gazes = net({"face": img})

                            batch_size = min(len(gazes), len(names), len(gts))

                            for k in range(batch_size):
                                try:
                                    gaze_np = safe_tensor_to_numpy(gazes[k])
                                    gt_np = safe_tensor_to_numpy(gts[k])

                                    if gaze_np is None or gt_np is None:
                                        continue

                                    gaze_3d = gazeto3d(gaze_np)
                                    gt_3d = gazeto3d(gt_np)

                                    if gaze_3d is not None and gt_3d is not None:
                                        error = angular(gaze_3d, gt_3d)
                                        if error is not None and not np.isnan(error):
                                            count += 1
                                            accs += error
                                            all_errors.append(error)

                                            # 保存图像数据
                                            image_data = safe_tensor_to_numpy(data["face"][k])
                                            if image_data is not None:
                                                all_data.append((image_data, gaze_np, gt_np, names[k], error))

                                except Exception as e:
                                    print(f"Error processing sample {k} in batch {j}: {e}")
                                    continue

                            # 限制处理的批次数量
                            if len(all_data) >= max_visualize * 3:
                                print(f"Collected {len(all_data)} samples, breaking early")
                                break

                        except Exception as e:
                            print(f"Error processing batch {j}: {e}")
                            continue

                if all_errors and count > 0:
                    avg_error = accs / count
                    print(f"[{saveiter}] Total Samples: {count}, Average Error: {avg_error:.3f}°")

                    # 创建误差分布图
                    print("Creating error distribution plot...")
                    create_error_distribution_plot(all_errors, [data[3] for data in all_data], iter_save_path)

                    # 选择代表性样本进行详细可视化
                    if all_data:
                        print("Selecting representative samples...")
                        selected_samples = select_representative_samples(all_data, save_best_worst)

                        if selected_samples:
                            selected_samples = selected_samples[:max_visualize]  # 限制总数
                            print(f"Creating detailed visualizations for {len(selected_samples)} selected samples...")

                            # 为选定样本创建详细可视化
                            success_3d = 0
                            success_2d = 0
                            for i, (image, gaze_pred, gaze_gt, name, error) in enumerate(selected_samples):
                                try:
                                    print(f"Processing sample {i + 1}/{len(selected_samples)}: {name}")
                                    if i < 10:  # 只为前10个样本创建完整的可视化
                                        if visualize_enhanced_3d_gaze(gaze_pred, gaze_gt, name, iter_save_path, error):
                                            success_3d += 1
                                        if visualize_gaze_on_image_enhanced(image, gaze_pred, gaze_gt, name,
                                                                            iter_save_path):
                                            success_2d += 1
                                except Exception as e:
                                    print(f"Error creating detailed visualization for sample {i}: {e}")
                                    continue

                            # 创建对比网格
                            print("Creating comparison grid...")
                            create_comparison_grid(selected_samples, iter_save_path)

                            # 保存详细统计信息
                            print("Saving detailed statistics...")
                            try:
                                stats_file = os.path.join(iter_save_path, "statistics", "detailed_stats.txt")
                                ensure_dir_exists(os.path.dirname(stats_file))
                                with open(stats_file, 'w') as f:
                                    f.write(f"Iteration {saveiter} Statistics\n")
                                    f.write(f"=" * 50 + "\n")
                                    f.write(f"Total Samples: {count}\n")
                                    f.write(f"Average Error: {avg_error:.4f}°\n")
                                    f.write(f"Median Error: {np.median(all_errors):.4f}°\n")
                                    f.write(f"Standard Deviation: {np.std(all_errors):.4f}°\n")
                                    f.write(f"Min Error: {np.min(all_errors):.4f}°\n")
                                    f.write(f"Max Error: {np.max(all_errors):.4f}°\n")
                                    f.write(f"50th Percentile: {np.percentile(all_errors, 50):.4f}°\n")
                                    f.write(f"95th Percentile: {np.percentile(all_errors, 95):.4f}°\n")
                                    f.write(f"99th Percentile: {np.percentile(all_errors, 99):.4f}°\n")
                                    f.write(f"\nVisualization Success:\n")
                                    f.write(f"3D visualizations: {success_3d}/{min(10, len(selected_samples))}\n")
                                    f.write(f"2D visualizations: {success_2d}/{min(10, len(selected_samples))}\n")
                                    f.write(f"\nQuality Distribution:\n")
                                    f.write(
                                        f"Excellent (<5°): {np.sum(np.array(all_errors) < 5)}/{len(all_errors)} ({np.sum(np.array(all_errors) < 5) / len(all_errors) * 100:.1f}%)\n")
                                    f.write(
                                        f"Good (<10°): {np.sum(np.array(all_errors) < 10)}/{len(all_errors)} ({np.sum(np.array(all_errors) < 10) / len(all_errors) * 100:.1f}%)\n")
                                    f.write(
                                        f"Acceptable (<15°): {np.sum(np.array(all_errors) < 15)}/{len(all_errors)} ({np.sum(np.array(all_errors) < 15) / len(all_errors) * 100:.1f}%)\n")
                                    f.write(
                                        f"Poor (≥15°): {np.sum(np.array(all_errors) >= 15)}/{len(all_errors)} ({np.sum(np.array(all_errors) >= 15) / len(all_errors) * 100:.1f}%)\n")
                                print(f"Saved statistics to: {stats_file}")
                            except Exception as e:
                                print(f"Error saving statistics: {e}")

                            print(f"Completed visualization for iteration {saveiter}")
                            print(f"Results saved to: {iter_save_path}")
                            print(f"Successfully created {success_3d} 3D and {success_2d} 2D visualizations")
                        else:
                            print(f"No samples selected for iteration {saveiter}")
                    else:
                        print(f"No valid data collected for iteration {saveiter}")
                else:
                    print(f"No valid errors calculated for iteration {saveiter}")

            except Exception as e:
                print(f"Error processing iteration {saveiter}: {e}")
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"Error in visualization process: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # 可视化参数
        MAX_VISUALIZE = 20  # 最大详细可视化样本数
        SAVE_BEST_WORST = 5  # 保存最好/最坏样本数量

        print("Starting Enhanced Gaze Estimation Visualization")
        print(f"Max visualize samples: {MAX_VISUALIZE}")
        print(f"Best/worst samples to save: {SAVE_BEST_WORST}")

        visualize('/root/autodl-tmp/Gaze360/Traintest/config/config_mpii.yaml',
                  max_visualize=MAX_VISUALIZE,
                  save_best_worst=SAVE_BEST_WORST)

        print("Visualization completed successfully!")

    except Exception as e:
        print(f"Main execution error: {e}")
        traceback.print_exc()