import cv2
import numpy as np
import torch
import base64
import json
import os
import argparse
import time
from io import BytesIO
from PIL import Image
from codes.model.deepcrack import DeepCrack
from codes.config import Config as cfg

# 尝试导入opencv-contrib-python的额外模块
try:
    import cv2.ximgproc
    XIMGPROC_AVAILABLE = True
except ImportError:
    XIMGPROC_AVAILABLE = False
    print("警告: cv2.ximgproc 不可用，将使用替代的骨架化方法")

class DeepCrackInference:
    def __init__(self, model_path='codes/checkpoints/DeepCrack_CT260_FT1.pth', device='cuda', 
                 pixel_threshold=50, save_to_local=False):
        """
        初始化DeepCrack推理器
        
        Args:
            model_path (str): 预训练模型路径
            device (str): 设备类型 ('cuda' 或 'cpu')
            pixel_threshold (int): 裂缝像素数量阈值 (默认: 50)
            save_to_local (bool): 是否保存检测结果图到本地
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pixel_threshold = pixel_threshold
        self.save_to_local = save_to_local
        
        # 处理模型路径，支持相对路径
        if not os.path.isabs(model_path):
            # 如果是相对路径，尝试多个可能的位置
            possible_paths = [
                model_path,  # 当前目录
                os.path.join('..', model_path),  # 上级目录
                os.path.join('codes', model_path),  # codes目录
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            else:
                self.model_path = model_path
        else:
            self.model_path = model_path
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self):
        """加载预训练模型"""
        model = DeepCrack()
        
        # 检查是否有多个GPU
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        model.to(self.device)
        
        # 加载预训练权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"模型加载成功: {self.model_path}")
        else:
            print(f"警告: 模型文件不存在 {self.model_path}")
            print("使用随机初始化的模型")
        
        model.eval()
        return model
    
    def _preprocess_image(self, image_path):
        """
        预处理输入图像
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小到512x512
        img = cv2.resize(img, (512, 512))
        
        # 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        
        # 转换为张量 (C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
        
        # 添加batch维度 (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def _postprocess_output(self, output):
        """
        后处理模型输出
        
        Args:
            output (torch.Tensor): 模型输出
            
        Returns:
            numpy.ndarray: 处理后的二值化图像
        """
        # 应用sigmoid激活
        output = torch.sigmoid(output)
        
        # 转换为numpy数组
        output_np = output.squeeze().cpu().numpy()
        
        # 二值化 (阈值0.5)
        binary_output = (output_np > 0.5).astype(np.uint8)
        
        return binary_output
    
    def _image_to_base64(self, image_array):
        """
        将图像数组转换为base64字符串
        
        Args:
            image_array (numpy.ndarray): 图像数组
            
        Returns:
            str: base64编码的图像字符串
        """
        # 确保图像是uint8类型
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # 转换为PIL图像
        if len(image_array.shape) == 2:  # 灰度图
            pil_image = Image.fromarray(image_array)
        else:  # 彩色图
            pil_image = Image.fromarray(image_array)
        
        # 转换为base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def _detect_crack(self, binary_mask):
        """
        检测是否有裂缝
        
        Args:
            binary_mask (numpy.ndarray): 二值化掩码
            
        Returns:
            bool: 是否有裂缝
        """
        # 计算裂缝像素的数量
        crack_pixel_count = np.sum(binary_mask)
        
        return crack_pixel_count > self.pixel_threshold
    
    def _classify_crack_type(self, binary_mask):
        """
        对裂缝进行分类：正常、蜂窝麻面、墙体开裂、外观破损
        
        Args:
            binary_mask (numpy.ndarray): 二值化掩码
            
        Returns:
            str: 裂缝类型
        """
        # 计算裂缝像素总数
        crack_pixel_count = np.sum(binary_mask)
        
        # 如果没有裂缝，直接返回正常
        if crack_pixel_count <= self.pixel_threshold:
            return "正常"
        
        # 先膨胀再腐蚀（闭运算）来连接断裂的线条
        kernel = np.ones((3, 3), np.uint8)
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 统计连通组件数量（线条数）
        num_labels, labels = cv2.connectedComponents(closed_mask)
        line_count = num_labels - 1  # 减去背景
        
        # 根据线条数进行分类
        if line_count <= 40:  # 线条少 - 墙体开裂
            return "墙体开裂"
        elif line_count <= 200:  # 线条中等 - 外观破损
            return "外观破损"
        else:  # 线条多 - 蜂窝麻面
            return "蜂窝麻面"
    
    def _simple_skeletonize(self, binary_mask):
        """
        简单的骨架化实现（当cv2.ximgproc不可用时使用）
        
        Args:
            binary_mask (numpy.ndarray): 二值化掩码
            
        Returns:
            numpy.ndarray: 骨架化后的图像
        """
        # 使用形态学操作进行简单的骨架化
        kernel = np.ones((3, 3), np.uint8)
        
        # 迭代腐蚀直到无法继续
        skeleton = binary_mask.copy()
        while True:
            eroded = cv2.erode(skeleton, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(skeleton, temp)
            skeleton = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
                
        return skeleton
    
    def _save_detection_result(self, image_path, binary_mask, has_crack, crack_type):
        """
        保存检测结果到本地
        
        Args:
            image_path (str): 原始图像路径
            binary_mask (numpy.ndarray): 二值化掩码
            has_crack (bool): 是否有裂缝
            crack_type (str): 裂缝类型
        """
        try:
            # 创建保存目录
            save_dir = "detection_results"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 生成文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = int(time.time())
            
            # 保存原始掩码（黑底白线）
            mask_path = os.path.join(save_dir, f"{base_name}_mask_{timestamp}.png")
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_image.save(mask_path)
            
            # 保存可视化结果（白底黑线）
            vis_path = os.path.join(save_dir, f"{base_name}_visualization_{timestamp}.png")
            vis_image = np.ones((512, 512, 3), dtype=np.uint8) * 255  # 白色背景
            vis_image[binary_mask > 0.5] = [0, 0, 0]  # 黑色裂缝
            vis_image = Image.fromarray(vis_image)
            vis_image.save(vis_path)
            
            # 保存检测信息
            info_path = os.path.join(save_dir, f"{base_name}_info_{timestamp}.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"原始图像: {image_path}\n")
                f.write(f"检测结果: {'有裂缝' if has_crack else '无裂缝'}\n")
                f.write(f"裂缝类型: {crack_type}\n")
                f.write(f"裂缝像素数: {np.sum(binary_mask)}\n")
                f.write(f"像素阈值: {self.pixel_threshold}\n")
                f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"[INFO] 检测结果已保存到 {save_dir}/")
            print(f"  - 掩码图像: {mask_path}")
            print(f"  - 可视化图像: {vis_path}")
            print(f"  - 检测信息: {info_path}")
            
        except Exception as e:
            print(f"[WARNING] 保存检测结果失败: {e}")
    
    def predict(self, image_path):
        """
        对单张图像进行裂缝检测和分类
        
        Args:
            image_path (str): 输入图像路径
            
        Returns:
            dict: 包含base64图像、检测结果和分类结果的字典
        """
        try:
            # 预处理图像
            input_tensor = self._preprocess_image(image_path)
            
            # 模型推理
            with torch.no_grad():
                output, _, _, _, _, _ = self.model(input_tensor)
            
            # 后处理输出
            binary_mask = self._postprocess_output(output)
            
            # 检测是否有裂缝
            has_crack = self._detect_crack(binary_mask)
            
            # 分类裂缝类型
            crack_type = self._classify_crack_type(binary_mask)
            
            # 转换为base64
            mask_base64 = self._image_to_base64(binary_mask)
            
            # 如果设置了保存到本地，则保存检测结果图
            if self.save_to_local:
                self._save_detection_result(image_path, binary_mask, has_crack, crack_type)
            
            # 构建结果
            result = {
                "has_crack": bool(has_crack),
                "crack_type": str(crack_type),
                "crack_mask_base64": str(mask_base64)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e)
            }


def main():
    parser = argparse.ArgumentParser(description='DeepCrack单张图像推理')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, default='codes/checkpoints/DeepCrack_CT260_FT1.pth', 
                       help='预训练模型路径')
    parser.add_argument('--pixel_threshold', type=int, default=50, 
                       help='裂缝像素数量阈值 (默认: 50)')
    parser.add_argument('--save_to_local', action='store_true', 
                       help='是否保存检测结果图到本地')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='设备类型 (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None, 
                       help='输出JSON文件路径 (可选)')
    
    args = parser.parse_args()
    
    # 检查输入图像是否存在
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在 {args.image_path}")
        return
    
    # 创建推理器
    try:
        inference = DeepCrackInference(
            model_path=args.model_path, 
            device=args.device,
            pixel_threshold=args.pixel_threshold,
            save_to_local=args.save_to_local
        )
    except Exception as e:
        print(f"错误: 无法初始化推理器 - {e}")
        return
    
    # 进行推理
    result = inference.predict(args.image_path)
    
    # 输出结果
    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(result_json)

    #  保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result_json)
        print(f"结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
