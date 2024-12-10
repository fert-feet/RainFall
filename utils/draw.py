# -*- coding: utf-8 -*-
# @Time : 2023/2/22 15:07 
# @Author : Mingzheng 
# @File : draw.py
# @desc :
import os


import matplotlib.pyplot as plt




class Plotter:
    def __init__(self, labels, outputs, R2, RMSE, MSE, MAE, feature_name):
        self.labels = labels
        self.outputs = outputs
        self.R2 = R2
        self.RMSE = RMSE
        self.MSE = MSE
        self.MAE = MAE
        self.feature_name = feature_name
        self.img_save_path = "./data/img"

    def simply_draw(self):
        plt.figure(figsize=(14, 7))

        # 绘制 R2 曲线
        plt.plot(self.R2, label='R2', color='#1f77b4', linestyle='-', linewidth=1.5, marker='o', markersize=3)
        plt.annotate(f'{self.R2[-1]:.4f}', xy=(len(self.R2) - 1, self.R2[-1]), xytext=(len(self.R2) - 1, self.R2[-1] + 0.01),
                     textcoords='data', fontsize=10, ha='center', va='bottom')

        # 绘制 MSE 曲线
        plt.plot(self.MSE, label='MSE', color='#ff7f0e', linestyle='--', linewidth=1.5, marker='s', markersize=3)
        plt.annotate(f'{self.MSE[-1]:.4f}', xy=(len(self.MSE) - 1, self.MSE[-1]), xytext=(len(self.MSE) - 1, self.MSE[-1] + 0.01),
                     textcoords='data', fontsize=10, ha='center', va='bottom')

        # 绘制 RMSE 曲线
        plt.plot(self.RMSE, label='RMSE', color='#2ca02c', linestyle='-.', linewidth=1.5, marker='^', markersize=3)
        plt.annotate(f'{self.RMSE[-1]:.4f}', xy=(len(self.RMSE) - 1, self.RMSE[-1]), xytext=(len(self.RMSE) - 1, self.RMSE[-1] + 0.01),
                     textcoords='data', fontsize=10, ha='center', va='bottom')

        # 绘制 MAE 曲线
        plt.plot(self.MAE, label='MAE', color='#d62728', linestyle=':', linewidth=1.5, marker='x', markersize=3)
        plt.annotate(f'{self.MAE[-1]:.4f}', xy=(len(self.MAE) - 1, self.MAE[-1]), xytext=(len(self.MAE) - 1, self.MAE[-1] + 0.01),
                     textcoords='data', fontsize=10, ha='center', va='bottom')

        # 添加图例
        plt.legend(fontsize=12)

        # 添加标题和标签
        plt.title('Metrics over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)

        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(self.img_save_path, f'{self.feature_name}_metrics.png'), dpi=300)

    def plot_outputs_scatter(self):
        plt.figure(figsize=(10, 6))

        # 绘制输出值的散点图
        plt.scatter(range(len(self.outputs)), self.outputs, color='#1f77b4', label='Outputs', alpha=0.7)

        # 绘制标签值的散点图
        plt.scatter(range(len(self.labels)), self.labels, color='#ff7f0e', label='Labels', alpha=0.7)

        # 添加图例
        plt.legend(fontsize=12)

        # 添加标题和标签
        plt.title('Outputs and Labels Scatter Distribution', fontsize=16)
        plt.xlabel('Index', fontsize=14)
        plt.ylabel('Value', fontsize=14)

        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(self.img_save_path, f'{self.feature_name}_outputs_labels_scatter.png'), dpi=300)

    def plot_outputs_vs_labels(self):
        plt.figure(figsize=(14, 7))

        # 绘制输出值的折线图
        plt.plot(self.outputs[::100], color='#1f77b4', label='Outputs', linewidth=1.5)

        # 绘制标签值的折线图
        plt.plot(self.labels[::100], color='#ff7f0e', label='Labels', linewidth=1.5)

        # 添加图例
        plt.legend(fontsize=12)

        # 添加标题和标签
        plt.title('Outputs vs Labels', fontsize=16)
        plt.xlabel('Index', fontsize=14)
        plt.ylabel('Value', fontsize=14)

        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(self.img_save_path, f'{self.feature_name}_outputs_labels_plot.png'), dpi=300)
