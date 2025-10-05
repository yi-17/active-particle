# -*- coding: utf-8 -*-
import cv2
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox, filedialog
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import platform

class ParticleAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("粒子分析软件")
        
        # 初始化变量
        self.video_counter = 0
        self.current_video_path = None
        
        # 初始化子文件夹名称
        self.subfolder_names = {
            "采集图片合成视频": "image_video_collection",
            "单一粒子蓝色识别MSD": "single_blue_particle_msd",
            "单一粒子绿色识别MSD": "single_green_particle_msd",
            "单一粒子红色识别MSD": "single_red_particle_msd",
            "群体识别得到对关联函数图像": "group_correlation_analysis"
        }
        
        # 创建输出目录
        self.output_dir = self.create_output_directory()
        
        # 创建界面元素
        self.create_widgets()
        
    def create_output_directory(self):
        base_dir = "analysis_results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"Analysis_{timestamp}")
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子文件夹
        for folder_name in self.subfolder_names.values():
            os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
        
        return output_dir
    
    def create_widgets(self):
        # 创建五个功能按钮
        btn_collect = tk.Button(self.root, text="采集图片合成视频", command=self.collect_images_and_create_video)
        btn_collect.pack(fill=tk.X, padx=20, pady=10)
        
        btn_blue_msd = tk.Button(self.root, text="单一粒子蓝色识别MSD", command=self.single_particle_blue_msd)
        btn_blue_msd.pack(fill=tk.X, padx=20, pady=10)
        
        btn_green_msd = tk.Button(self.root, text="单一粒子绿色识别MSD", command=self.single_particle_green_msd)
        btn_green_msd.pack(fill=tk.X, padx=20, pady=10)
        
        btn_red_msd = tk.Button(self.root, text="单一粒子红色识别MSD", command=self.single_particle_red_msd)
        btn_red_msd.pack(fill=tk.X, padx=20, pady=10)
        
        btn_group_corr = tk.Button(self.root, text="群体识别得到对关联函数图像", command=self.group_particles_correlation)
        btn_group_corr.pack(fill=tk.X, padx=20, pady=10)
        
        btn_view_files = tk.Button(self.root, text="查看结果文件", command=self.view_result_files)
        btn_view_files.pack(fill=tk.X, padx=20, pady=10)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update()
    
    def collect_images_and_create_video(self):
        self.update_status("开始采集图片...")
        
        # 创建保存目录
        save_subdir = self.subfolder_names["采集图片合成视频"]
        save_dir = os.path.join(self.output_dir, save_subdir)
        os.makedirs(save_dir, exist_ok=True)
        
        # 采集前清空旧图片（只删 image_*.jpg，不删视频）
        for fname in os.listdir(save_dir):
            if fname.startswith("image_") and fname.endswith(".jpg"):
                os.remove(os.path.join(save_dir, fname))
        
        # 标记当前采集会话
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头！")
            self.update_status("错误：无法打开摄像头")
            return
        
        # 采集图片（编号保存，每张做圆台掩膜）
        image_count = 500
        h, w = None, None
        center, radius = None, None
        for i in tqdm(range(image_count), desc="拍摄图片进度"):
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("错误", "无法读取帧！")
                break
            if h is None or w is None:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2 - 40)
                radius = min(w, h) // 2 + 100
            # 创建圆形掩膜
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            frame_masked = frame.copy()
            frame_masked[mask == 0] = 0
            # 按编号保存图片
            image_path = os.path.join(save_dir, f"image_{i:04d}.jpg")
            cv2.imwrite(image_path, frame_masked)
            time.sleep(0.04)
        cap.release()
        cv2.destroyAllWindows()
        self.update_status(f"已采集 {image_count} 张图片")
        
        # 合成视频（按编号顺序读取图片，生成新视频文件）
        video_name = f"captured_video_{timestamp}.mp4"
        video_path = os.path.join(save_dir, video_name)
        images = [img for img in os.listdir(save_dir) if img.startswith("image_") and img.endswith(".jpg")]
        images.sort()
        if not images:
            messagebox.showerror("错误", "没有找到图片文件！")
            return
        frame = cv2.imread(os.path.join(save_dir, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for image in tqdm(images, desc="合成视频进度"):
            video.write(cv2.imread(os.path.join(save_dir, image)))
        video.release()
        self.update_status(f"视频文件已保存为：{video_path}")
        messagebox.showinfo("完成", f"图片采集和视频合成完成！\n视频保存为：{video_path}")
    
    def select_video(self):
        self.update_status("请选择视频文件...")
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov"), ("所有文件", "*.*")]
        )
        if file_path:
            self.current_video_path = file_path
            self.update_status(f"已选择视频：{file_path}")
            return True
        else:
            self.update_status("未选择视频")
            return False
    
    def single_particle_blue_msd(self):
        if not self.select_video():
            return
        self.update_status("开始蓝色粒子轨迹分析...")
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        records = []
        traj_points = []
        valid_count = 0
        h, w = None, None
        center, radius = None, None
        for frame_id in tqdm(range(frame_count), desc="处理视频进度"):
            ret, frame = cap.read()
            if not ret:
                break
            if h is None or w is None:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2 - 40)
                radius = min(w, h) // 2 + 100
            # 创建圆形掩膜
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            frame_masked = frame.copy()
            frame_masked[mask == 0] = 0
            # 检测轨迹前统一用掩膜处理后的frame_masked
            t = frame_id / fps
            x, y = self.detect_white_dot_on_blue(frame_masked)
            records.append({'t': t, 'x': x, 'y': y})
            if x is not None and y is not None:
                traj_points.append((x, y))
                valid_count += 1
                cv2.circle(frame_masked, (x, y), 5, (255, 0, 0), -1)
            # 优化窗口显示，窗口为视频实际分辨率的80%，保证完整显示
            scale = 0.8
            display_w, display_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(frame_masked, (display_w, display_h))
            cv2.namedWindow('Blue_Result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Blue_Result', display_w, display_h)
            cv2.imshow('Blue_Result', display_frame)
        cap.release()
        cv2.destroyAllWindows()
        self.update_status(f"已处理 {frame_count} 帧，识别到点的帧数：{valid_count}")
        messagebox.showinfo("识别率", f"总帧数：{frame_count}\n识别到点的帧数：{valid_count}\n识别率：{valid_count/frame_count*100:.2f}%")
        df = pd.DataFrame(records)
        output_subdir = self.subfolder_names["单一粒子蓝色识别MSD"]
        output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = os.path.join(output_dir, f"blue_particle_trajectory_{timestamp}.xlsx")
        df.to_excel(output_excel, index=False)
        output_png = os.path.join(output_dir, f"blue_particle_msd_{timestamp}.png")
        self.plot_msd_from_df(df, output_png)
        output_traj_png = os.path.join(output_dir, f"blue_particle_trajectory_{timestamp}.png")
        self.plot_corrected_2d_trajectory(traj_points, output_traj_png)
        messagebox.showinfo("完成", f"蓝色粒子分析完成！\n轨迹数据已保存为：{output_excel}\nMSD图像已保存为：{output_png}\n轨迹图已保存为：{output_traj_png}")
    
    def single_particle_green_msd(self):
        if not self.select_video():
            return
        self.update_status("开始绿色粒子轨迹分析...")
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        records = []
        traj_points = []
        valid_count = 0
        h, w = None, None
        center, radius = None, None
        for frame_id in tqdm(range(frame_count), desc="处理视频进度"):
            ret, frame = cap.read()
            if not ret:
                break
            if h is None or w is None:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2 - 40)
                radius = min(w, h) // 2 + 100
            # 创建圆形掩膜
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            frame_masked = frame.copy()
            frame_masked[mask == 0] = 0
            # 检测轨迹前统一用掩膜处理后的frame_masked
            t = frame_id / fps
            x, y = self.detect_white_dot_on_green(frame_masked)
            records.append({'t': t, 'x': x, 'y': y})
            if x is not None and y is not None:
                traj_points.append((x, y))
                valid_count += 1
                cv2.circle(frame_masked, (x, y), 5, (0, 255, 0), -1)
            # 优化窗口显示，窗口为视频实际分辨率的80%，保证完整显示
            scale = 0.8
            display_w, display_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(frame_masked, (display_w, display_h))
            cv2.namedWindow('Green_Result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Green_Result', display_w, display_h)
            cv2.imshow('Green_Result', display_frame)
        cap.release()
        cv2.destroyAllWindows()
        self.update_status(f"已处理 {frame_count} 帧，识别到点的帧数：{valid_count}")
        messagebox.showinfo("识别率", f"总帧数：{frame_count}\n识别到点的帧数：{valid_count}\n识别率：{valid_count/frame_count*100:.2f}%")
        df = pd.DataFrame(records)
        output_subdir = self.subfolder_names["单一粒子绿色识别MSD"]
        output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = os.path.join(output_dir, f"green_particle_trajectory_{timestamp}.xlsx")
        df.to_excel(output_excel, index=False)
        output_png = os.path.join(output_dir, f"green_particle_msd_{timestamp}.png")
        self.plot_msd_from_df(df, output_png)
        output_traj_png = os.path.join(output_dir, f"green_particle_trajectory_{timestamp}.png")
        self.plot_corrected_2d_trajectory(traj_points, output_traj_png)
        messagebox.showinfo("完成", f"绿色粒子分析完成！\n轨迹数据已保存为：{output_excel}\nMSD图像已保存为：{output_png}\n轨迹图已保存为：{output_traj_png}")
    
    def single_particle_red_msd(self):
        if not self.select_video():
            return
        self.update_status("开始红色粒子轨迹分析...")
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        records = []
        traj_points = []
        valid_count = 0
        h, w = None, None
        center, radius = None, None
        for frame_id in tqdm(range(frame_count), desc="处理视频进度"):
            ret, frame = cap.read()
            if not ret:
                break
            if h is None or w is None:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2 - 40)
                radius = min(w, h) // 2 + 100
            # 创建圆形掩膜
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            frame_masked = frame.copy()
            frame_masked[mask == 0] = 0
            # 检测轨迹前统一用掩膜处理后的frame_masked
            t = frame_id / fps
            x, y = self.detect_white_dot_on_red(frame_masked)
            records.append({'t': t, 'x': x, 'y': y})
            if x is not None and y is not None:
                traj_points.append((x, y))
                valid_count += 1
                cv2.circle(frame_masked, (x, y), 5, (0, 0, 255), -1)
            # 优化窗口显示，窗口为视频实际分辨率的80%，保证完整显示
            scale = 0.8
            display_w, display_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(frame_masked, (display_w, display_h))
            cv2.namedWindow('Red_Result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Red_Result', display_w, display_h)
            cv2.imshow('Red_Result', display_frame)
        cap.release()
        cv2.destroyAllWindows()
        self.update_status(f"已处理 {frame_count} 帧，识别到点的帧数：{valid_count}")
        messagebox.showinfo("识别率", f"总帧数：{frame_count}\n识别到点的帧数：{valid_count}\n识别率：{valid_count/frame_count*100:.2f}%")
        df = pd.DataFrame(records)
        output_subdir = self.subfolder_names["单一粒子红色识别MSD"]
        output_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = os.path.join(output_dir, f"red_particle_trajectory_{timestamp}.xlsx")
        df.to_excel(output_excel, index=False)
        output_png = os.path.join(output_dir, f"red_particle_msd_{timestamp}.png")
        self.plot_msd_from_df(df, output_png)
        output_traj_png = os.path.join(output_dir, f"red_particle_trajectory_{timestamp}.png")
        self.plot_corrected_2d_trajectory(traj_points, output_traj_png)
        messagebox.showinfo("完成", f"红色粒子分析完成！\n轨迹数据已保存为：{output_excel}\nMSD图像已保存为：{output_png}\n轨迹图已保存为：{output_traj_png}")
    
    def detect_white_dot_on_blue(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 80, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cv2.imshow('Blue_Mask', blue_mask)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.imshow('Blue_White_Mask', np.zeros_like(blue_mask))
            cv2.waitKey(1)
            return None, None
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(blue_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        blue_area = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('Blue_Area', blue_area)
        blue_hsv = cv2.cvtColor(blue_area, cv2.COLOR_BGR2HSV)
        v = blue_hsv[:,:,2]
        cv2.imshow('Blue_Area_V', v)
        # Otsu自动阈值分割亮区
        _, white_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        # 只考虑面积大于100的最大亮区
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_largest = None
        max_area = 0
        for cnt in white_contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_area:
                max_area = area
                white_largest = cnt
        mask_with_center = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        cx, cy = None, None
        if white_largest is not None:
            M = cv2.moments(white_largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(mask_with_center, (cx, cy), 5, (255, 0, 0), -1)
        cv2.imshow('Blue_White_Mask', mask_with_center)
        cv2.waitKey(1)
        if cx is None or cy is None:
            return None, None
        return cx, cy
    
    def detect_white_dot_on_green(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 放宽绿色HSV容差
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        cv2.imshow('Green_Mask', green_mask)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.imshow('Green_White_Mask', np.zeros_like(green_mask))
            cv2.waitKey(1)
            return None, None
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(green_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        green_area = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('Green_Area', green_area)
        green_hsv = cv2.cvtColor(green_area, cv2.COLOR_BGR2HSV)
        v = green_hsv[:,:,2]
        cv2.imshow('Green_Area_V', v)
        # Otsu自动阈值分割亮区
        _, white_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        # 只考虑面积大于50的最大亮区
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_largest = None
        max_area = 0
        for cnt in white_contours:
            area = cv2.contourArea(cnt)
            if area > 50 and area > max_area:
                max_area = area
                white_largest = cnt
        mask_with_center = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        cx, cy = None, None
        if white_largest is not None:
            M = cv2.moments(white_largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(mask_with_center, (cx, cy), 5, (0, 255, 0), -1)
        cv2.imshow('Green_White_Mask', mask_with_center)
        cv2.waitKey(1)
        if cx is None or cy is None:
            return None, None
        return cx, cy
    
    def detect_white_dot_on_red(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        cv2.imshow('Red_Mask', red_mask)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cv2.imshow('Red_White_Mask', np.zeros_like(red_mask))
            cv2.waitKey(1)
            return None, None
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(red_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        red_area = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('Red_Area', red_area)
        red_hsv = cv2.cvtColor(red_area, cv2.COLOR_BGR2HSV)
        v = red_hsv[:,:,2]
        cv2.imshow('Red_Area_V', v)
        # Otsu自动阈值分割亮区
        _, white_mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        # 只考虑面积大于100的最大亮区
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_largest = None
        max_area = 0
        for cnt in white_contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_area:
                max_area = area
                white_largest = cnt
        mask_with_center = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        cx, cy = None, None
        if white_largest is not None:
            M = cv2.moments(white_largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(mask_with_center, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow('Red_White_Mask', mask_with_center)
        cv2.waitKey(1)
        if cx is None or cy is None:
            return None, None
        return cx, cy
    
    def plot_msd_from_df(self, df, output_png):
        df_valid = df.dropna(subset=['x', 'y'])
        t = df_valid['t'].values
        x = df_valid['x'].values
        y = df_valid['y'].values
        if len(t) < 2:
            # 有效点太少，无法计算MSD
            messagebox.showwarning("提示", "有效轨迹点太少，无法绘制MSD曲线")
            return
        max_lag = min(len(t) - 1, 100)
        msd = np.zeros(max_lag)
        time_lags = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            displacement_squared = dx**2 + dy**2
            msd[lag - 1] = np.mean(displacement_squared)
            time_lags[lag - 1] = np.mean(t[lag:] - t[:-lag])
        valid = (time_lags > 0) & (msd > 0)
        log_time_lags = np.log(time_lags[valid])
        log_msd = np.log(msd[valid])
        slope, intercept = np.polyfit(log_time_lags, log_msd, 1)
        fit_line = np.exp(intercept) * time_lags[valid]**slope
        plt.figure(figsize=(8, 6))
        plt.plot(time_lags[valid], msd[valid], 'o', color='r', label='MSD')
        plt.plot(time_lags[valid], fit_line, '-', color='b', label=f'Fit: slope={slope:.3f}')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('MSD vs Time Lag (Log-Log)', fontsize=13)
        plt.xlabel('Time Lag (log)', fontsize=11)
        plt.ylabel('MSD (log)', fontsize=11)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
    
    def plot_corrected_2d_trajectory(self, traj_points, output_png):
        if not traj_points:
            return
            
        traj_points = np.array(traj_points)
        
        plt.figure(figsize=(10, 8))
        plt.plot(traj_points[:, 0], traj_points[:, 1], '-', color='b', linewidth=1)
        plt.scatter(traj_points[:, 0], traj_points[:, 1], c='r', s=10)
        plt.title('2D Particle Trajectory', fontsize=14)
        plt.xlabel('X Position', fontsize=12)
        plt.ylabel('Y Position', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
    
    def group_particles_correlation(self):
        if not self.select_video():
            return
        
        self.update_status("开始群体粒子分析...")
        
        analyzer = ParticleAnalyzer(self.current_video_path, output_dir=self.output_dir)
        positions, r_values, avg_g_r = analyzer.analyze_video()
        
        if r_values is not None and avg_g_r is not None:
            output_subdir = self.subfolder_names["群体识别得到对关联函数图像"]
            output_dir = os.path.join(self.output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_png = os.path.join(output_dir, f"group_particles_correlation_{timestamp}.png")
            self.plot_correlation_function(r_values, avg_g_r, output_png)
            messagebox.showinfo("完成", f"群体粒子分析完成！\n对关联函数图像已保存为：{output_png}")
    
    def plot_correlation_function(self, r, g_r, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(r, g_r, 'b-', label='g(r)')
        plt.xlabel('r (pixels)', fontsize=12)
        plt.ylabel('g(r)', fontsize=12)
        plt.title('Pair Correlation Function', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(filename)
        plt.close()
    
    def view_result_files(self):
        if not os.path.exists(self.output_dir):
            messagebox.showinfo("提示", "没有生成结果文件")
            return
        
        # 在不同的操作系统中打开文件夹的方式不同
        if platform.system() == "Windows":
            os.startfile(self.output_dir)
        elif platform.system() == "Darwin":  # macOS
            os.system(f"open '{self.output_dir}'")
        else:  # Linux
            os.system(f"xdg-open '{self.output_dir}'")

class ParticleAnalyzer:
    def __init__(self, video_path, output_dir='results', sample_interval=30):
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.time_interval = sample_interval / self.fps
    
    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=30,
            param2=12,
            minRadius=8,
            maxRadius=25
        )
        
        particles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_mask = np.zeros(gray.shape, dtype=np.uint8)
            
            for circle in circles[0, :]:
                x, y, r = circle
                
                check_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(check_mask, (x, y), r+5, 255, -1)
                if np.any(np.logical_and(detected_mask, check_mask)):
                    continue
                    
                ring_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(ring_mask, (x, y), r, 255, 2)
                
                ring_pixels = hsv[ring_mask > 0]
                if len(ring_pixels) > 0:
                    mean_hsv = np.mean(ring_pixels, axis=0)
                    std_hsv = np.std(ring_pixels, axis=0)
                    
                    is_blue = (80 <= mean_hsv[0] <= 150 and
                             mean_hsv[1] >= 20 and
                             mean_hsv[2] >= 20 and
                             std_hsv[0] < 30 and
                             std_hsv[1] > 3)
                    
                    if is_blue:
                        precise_ring_mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.circle(precise_ring_mask, (x, y), r, 255, 2)
                        
                        ring_coords = np.where(precise_ring_mask > 0)
                        ring_intensities = gray[ring_coords]
                        
                        if len(ring_intensities) > 0:
                            center_y = np.average(ring_coords[0], weights=ring_intensities)
                            center_x = np.average(ring_coords[1], weights=ring_intensities)
                            
                            if (abs(center_x - x) <= r/2 and abs(center_y - y) <= r/2):
                                particles.append([int(center_x), int(center_y)])
                                cv2.circle(detected_mask, (x, y), r, 255, -1)
        
        return np.array(particles)
    
    def calculate_pair_correlation(self, positions, bins=50, r_max=None):
        if len(positions) < 2:
            return None, None
            
        distances = pdist(positions)
        
        if r_max is None:
            r_max = np.max(distances) / 2
            
        hist, bin_edges = np.histogram(distances, bins=bins, range=(0, r_max))
        
        r = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        area = np.prod(np.ptp(positions, axis=0))
        density = len(positions) / area
        
        ring_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        
        g_r = hist / (ring_areas * density * len(positions))
        
        return r, g_r
    
    def analyze_video(self):
        frame_idx = 0
        all_positions = []
        all_g_r = []
        r_values = None
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.sample_interval == 0:
                positions = self.process_frame(frame)
                
                if len(positions) > 0:
                    all_positions.append(positions)
                    
                    r, g_r = self.calculate_pair_correlation(positions)
                    if r is not None:
                        if r_values is None:
                            r_values = r
                        all_g_r.append(g_r)
                
            frame_idx += 1
            
        self.cap.release()
        
        avg_g_r = np.mean(all_g_r, axis=0) if all_g_r else None
        
        return all_positions, r_values, avg_g_r

if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleAnalysisApp(root)
    root.mainloop()