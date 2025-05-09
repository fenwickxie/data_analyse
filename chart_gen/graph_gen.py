#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: fenwickxie
date: 2025-04-22 14:10:06
filename: graph_gen.py
version: 1.0
"""


import pandas as pd


class GraphGen:
    def __init__(self, metrics: dict[str, pd.DataFrame]):

        self.data = metrics

    def generate_graph(self, horizontal_axis="throttle", vertical_axis="latency"):

        pass

    def save_graph(self, filename):
        # 保存图表的逻辑
        pass


if __name__ == "__main__":
    pass