#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: fenwickxie
date: 2025-04-22 14:10:06
filename: graph_gen.py
version: 1.0
"""


import os
import pandas as pd
from matplotlib import pyplot as plt


class GraphGen:
    """
    GraphGen类用于对给定的指标数据和阶段数据进行可视化，支持生成柱状图和折线图，并可将图像保存到指定路径。
    参数:
        metrics (dict[str, pd.DataFrame]): 指标数据字典，键为指标名称，值为对应的pandas DataFrame。
        stages (list[pd.DataFrame], 可选): 阶段数据列表，每个元素为一个pandas DataFrame，默认为None。
        save_path (str, 可选): 图像保存路径，默认为"./chart_gen/graphs/"。
    方法:
        metric_visualize(horizontal_axis="throttle", vertical_axis=["LWSAngle", "monitor_diff", "slip_WhlSpd"], bar_width=0.2):
            对metrics中的每个DataFrame生成柱状图。横轴为horizontal_axis指定的列，纵轴为vertical_axis指定的多个列，支持多组数据并列显示。图像保存到save_path路径下。
        stage_visualize(horizontal_axis="timestamps", vertical_axis=[...]):
            对stages中的每个DataFrame生成折线图。横轴为horizontal_axis指定的列，纵轴为vertical_axis指定的多个列。每个DataFrame绘制在单独的图形上，并显示图像。
    用法示例:
        graph_gen = GraphGen(metrics, stages)
        graph_gen.metric_visualize()
        graph_gen.stage_visualize()
    """

    def __init__(
        self,
        metrics: dict[str, pd.DataFrame],
        stages: list[pd.DataFrame]=None,
        save_path: str = "./chart_gen/graphs/",
    ):

        # 初始化函数，接收一个字典类型的参数metrics和一个字符串类型的参数save_path
        self.metrics = metrics
        self.stages = stages
        # 将metrics赋值给self.data
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 如果save_path不存在，则创建该路径
        self.save_path = save_path

    def metric_visualize(
        self,
        horizontal_axis="throttle",
        vertical_axis=["LWSAngle", "monitor_diff", "slip_WhlSpd"],
        bar_width=0.2,
    ):

        # 遍历字典的键和值，每个键代表一个图表，每个值代表一个数据框
        for metric, df in self.metrics.items():
            if df.empty:
                continue

            max_value = df.groupby(horizontal_axis).max().reindex()
            # 初始化图表
            fig, ax = plt.subplot(figsize=(10, 6))
            index = range(len(max_value[horizontal_axis]))

            # 绘制柱状图
            for i, vertical in enumerate(vertical_axis):
                bars = plt.bar(
                    [x + i * bar_width for x in index],
                    max_value[vertical],
                    bar_width,
                    label=vertical,
                )
                # 在每个条形上添加数据标签
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center',
                        va='bottom'
                    )

            # 设置图表标题和轴标签
            ax.set_title(metric)
            ax.set_xlabel(horizontal_axis)
            ax.set_ylabel("Value")
            ax.set_xticks([i + bar_width for i in index])
            ax.set_xticklabels(max_value[horizontal_axis])

            # 添加图例
            ax.legend()
            plt.savefig(os.path.join(self.save_path, f"{metric}.png"))
        plt.show()

    def stage_visualize(
        self,
        horizontal_axis="timestamps",
        vertical_axis=[
            "LWSAngle",
            "WhlSpdFL_122",
            "WhlSpdFR_122",
            "WhlSpdRL_122",
            "WhlSpdRR_122",
            "FMSpd_242",
            "RMSpd_250",
        ],
    ):
        """
        在self.stages中为每个DataFrame生成线图，将指定的垂直轴相对于水平轴可视化,self.stages中的每个DataFrame都绘制在一个单独的图形上。

        Parameters:
            horizontal_axis (str): 用作x轴的列名. Defaults to "timestamps".
            vertical_axis (list of str): y轴的列名列表。默认为预定义的列表。

        """
        # 根据传入self.stages生成折线图,self.stages是多个二维dataframe组成的列表，列表内单个元素是单个dataframe，这个dataframe的折线图绘制在一个图上
        if self.stages is not None:
            for idx, df in enumerate(self.stages):
                if df.empty:
                    continue
                plt.figure(figsize=(10, 3))
                for vertical in vertical_axis:
                    if vertical in df.columns and horizontal_axis in df.columns:
                        plt.plot(df[horizontal_axis], df[vertical], label=vertical)
                plt.title(getattr(df, "name", f"Stage {idx+1}"))
                plt.xlabel(horizontal_axis)
                plt.ylabel("Value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{self.save_path}/{idx}.png")
            plt.show()


if __name__ == "__main__":
    pass
