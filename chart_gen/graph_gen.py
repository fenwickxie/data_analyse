#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: fenwickxie
date: 2025-04-22 14:10:06
filename: graph_gen.py
version: 1.0
"""


import pandas as pd
from matplotlib import pyplot as plt


class GraphGen:
    def __init__(
        self,
        metrics: dict[str, pd.DataFrame],
        h_axis="throttle",
        v_axis=[
            "latency",
        ],
    ):

        self.data = metrics

    def generate_graph(self, horizontal_axis="throttle", vertical_axis="latency"):
        # 遍历字典的键和值，每个键代表一个图表，每个值代表一个数据框
        for metric, df in self.data.items():
            # 初始化图表
            fig = plt.figure(figsize=(10, 6))

            # 绘制图表
            plt.plot(
                x=df[horizontal_axis],
                y=df[vertical_axis],
                kind="line",
                ax=fig.add_subplot(111),
            )

            # 添加标题和标签
            fig.suptitle(f"{metric}")
            fig.subplots_adjust(top=0.9)
            fig.subplots_adjust(bottom=0.2)
            fig.subplots_adjust(left=0.2)
            fig.subplots_adjust(right=0.8)

            # 保存图表

    def save_graph(self, filename):
        # 保存图表的逻辑
        pass


if __name__ == "__main__":
    pass
