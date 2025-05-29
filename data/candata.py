#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: fenwickxie
date: 2025-04-22 14:04:31
filename: can_analysis.py
version: 1.0
"""

from itertools import product
import os
import numpy as np
import pandas as pd
import platform
from typing import TypeAlias, Union

StringPathLike: TypeAlias = Union[str, os.PathLike]

if platform.system() == "Windows":
    ENCODING = "gbk"
else:
    ENCODING = "utf-8"
R = 0.338

FRONT_MOTOR_TRANSMISSION_RATIO = 10.79
REAR_MOTOR_TRANSMISSION_RATIO = 10.81


class CanData:
    def __init__(self, data: os.PathLike | str):
        self.files = []
        if os.path.isfile(data):
            self.files.append(data)
            self.data = [pd.read_csv(data)]
        elif os.path.isdir(data):
            self.files = [
                os.path.join(data, f) for f in os.listdir(data) if f.endswith(".csv")
            ]
            self.data = [pd.read_csv(f) for f in self.files]

        self.grouped_files = self.__group_files_by_conditions()
        self.statics = pd.DataFrame()
        self.all_metrics = {}

    def __group_files_by_conditions(
        self,
        keywords: list = [["on", "off"], ["冰", "雪"], ["eco", "sport"]],
    ) -> dict[str, list]:
        """
        根据关键字列表自动生成条件组合，并对文件进行分组。

        Args:
            keywords (list): 关键字列表，用于生成分组条件。

        Returns:
            dict: 包含分组后的文件列表，格式为 {group_name: [file1, file2, ...]}。
        """

        # 生成条件组合
        conditions = {
            "_".join(combination): list(combination)
            for combination in product(*keywords)
        }

        grouped_files = {group_name: [] for group_name in conditions}

        for file in self.files:
            file_lower = os.path.basename(file).lower()  # 转为小写以便匹配
            for group_name, keywords in conditions.items():
                if all(keyword in file_lower for keyword in keywords):
                    grouped_files[group_name].append(file)
                    break  # 一个文件只归入一个分组

        return grouped_files

    def get_stage_idxs(
        self, data: pd.DataFrame, stage_filters: dict[str, tuple]
    ) -> list[tuple]:
        """
        根据输入参数筛选数据，支持不定数量的信号筛选，并获取从 min 增长到 max 的范围数据。

        Args:
            stage_filters (dict): 信号筛选条件字典，格式为 {signal_name: (min, max), ...}。

        Returns:
            slice_idx: 返回筛选后的数据段的起始和结束索引列表。
        """
        data["stage_marker"] = None  # 初始化阶段标记
        data["combined_flag"] = True  # 初始化综合标记

        start, end = data.index[0], data.index[-1]  # 获取数据段的起始和结束索引
        # 遍历信号筛选条件，逐个应用筛选
        for signal_name, value_range in stage_filters.items():
            min_val, max_val = value_range

            # 标记信号值是否从 min 开始逐渐增长到 max
            # data[f"{signal_name}_flag"] = False
            # in_growth_phase = False
            # for i in range(len(data)):
            #     if data[signal_name].iloc[i] == min_val:
            #         in_growth_phase = True
            #     if in_growth_phase and data[signal_name].iloc[i] >= max_val:
            #         in_growth_phase = False
            #     data[f"{signal_name}_flag"].iloc[i] = in_growth_phase
            # 标记信号值是否从 min 开始增长到 max，允许平台期（即值不降即可），但起步阶段不能有大段等于min_value的数据段
            data[f"{signal_name}_flag"] = False
            in_growth_phase = False
            growth_indices = []
            for i in range(start, end):
                _v = int(data[signal_name].iloc[i])
                if _v == min_val:
                    # 如果前一个也在增长阶段且也是min_val，则把前一个的flag改为False，避免大段min_val
                    if in_growth_phase and int(data.loc[i - 1, signal_name]) == min_val:
                        data.at[i - 1, f"{signal_name}_flag"] = False
                    in_growth_phase = True
                    data.at[i, f"{signal_name}_flag"] = True
                    growth_indices = [i]
                    continue
                if in_growth_phase:
                    # 允许平台期，只要不降即可，并且数值不能减小
                    # 允许小幅波动：如果下降幅度在允许范围内（如1单位），则不视为终止增长
                    if _v < data.loc[i - 1, signal_name]:
                        # 判断是否为小幅波动（如下降不超过1单位）
                        if data.loc[i - 1, signal_name] - _v <= 2:
                            data.at[i, f"{signal_name}_flag"] = True
                            growth_indices.append(i)
                            continue
                        # 否则视为终止增长阶段
                        for idx in growth_indices:
                            data.at[idx, f"{signal_name}_flag"] = False
                        in_growth_phase = False
                        growth_indices = []
                    elif _v >= min_val and _v <= max_val:  # 如果在范围内
                        data.at[i, f"{signal_name}_flag"] = True
                        growth_indices.append(i)
                        if _v == max_val:  # 如果达到最大值，完成一个有效增长阶段
                            in_growth_phase = False
                            growth_indices = []
                    elif _v > max_val:  # 如果超过max_val
                        # 只结束增长阶段，不改变之前的标记
                        in_growth_phase = False
                        growth_indices = []
            # 更新综合标记
            data["combined_flag"] &= data[f"{signal_name}_flag"]

        # 计算连续阶段标记
        data["stage_marker"] = (data["combined_flag"].diff().fillna(0) != 0).cumsum()

        # 筛选符合条件的数据段
        slice_idx_pd = (
            data[data["combined_flag"]]
            .groupby("stage_marker")
            .apply(lambda x: x.iloc[[0, -1]])
        )

        # 重置索引以便访问原始索引
        slice_idx_pd = slice_idx_pd.reset_index(drop=True)

        # 提取原始索引
        start_idx = slice_idx_pd["original_index"].iloc[::2].tolist()  # 起始索引
        end_idx = slice_idx_pd["original_index"].iloc[1::2].tolist()  # 结束索引

        # 将起始和结束索引组合成元组
        slice_idx = list(zip(start_idx, end_idx))

        return slice_idx

    def get_file_stage(self, data: pd.DataFrame, slice_idx) -> list[pd.DataFrame]:
        stages = []
        for start, end in slice_idx:
            period = data.loc[start:end]
            stages.append(period)

        return stages

    def get_statics(self, stage, signal_name):

        # 计算信号的平均值
        mean_value = stage[signal_name].mean()

        # 计算信号的标准差
        std_value = stage[signal_name].std()

        # 计算信号的最大值
        max_value = stage[signal_name].max()

        # 计算信号的最小值
        min_value = stage[signal_name].min()

        # 将结果添加到字典中
        self.statics[signal_name] = {
            "mean": mean_value,
            "std": std_value,
            "max": max_value,
            "min": min_value,
        }

        return self

    def get_start2end_change(
        self,
        stage,
        signal_names=[
            "LWSAngle_11F",
        ],
    ):
        metrics = {}
        # 计算信号始末变化量
        for signal_name in signal_names:
            metrics[f"{signal_name}_change"] = (
                stage.loc[stage.index[-1], signal_name]
                - stage.loc[stage.index[0], signal_name]
            )

        return metrics

    def get_slip_time(self, stage):

        # 计算信号首次变化时间
        stage["WhlSpdF"] = (stage["WhlSpdFL_122"] + stage["WhlSpdFR_122"]) / 2
        stage["WhlSpdR"] = (stage["WhlSpdRL_122"] + stage["WhlSpdRR_122"]) / 2
        stage["WhlSpd_diff"] = stage["WhlSpdR"] - stage["WhlSpdF"]
        slice_idx = self.get_stage_idxs(stage, {"WhlSpd_diff": (0, np.inf)})
        # 如果没有满足条件的数据，则返回None
        if len(slice_idx) == 0:
            return None
        closest_idx = (
            (
                stage["timestamps"]
                - (
                    (
                        stage.loc[slice_idx[0][0], "timestamps"]
                        + stage.loc[slice_idx[0][1], "timestamps"]
                    )
                    / 2
                )
            )
            .abs()
            .idxmin()
        )

        return closest_idx

    def get_slip_whlspd(
        self,
        stage,
        slip_time,
        whls=["WhlSpdFL_122", "WhlSpdFR_122", "WhlSpdRL_122", "WhlSpdRR_122"],
    ):

        stage_metrics = {}
        # # 根据get_slip_time方法计算的时间戳，获取对应的WhlSpd值
        # for whl in whls:
        #     # 获取对应的WhlSpd值
        #     whlspd_value = stage.loc[slip_time, whl]
        #     stage_metrics[f"slip_{whl}"] = whlspd_value
        stage_metrics[f"slip_WhlSpd"] = max(stage.loc[slip_time, whls].values)
        return stage_metrics

    def get_monitor_diff(self, stage, slip_time):
        """_summary_: 获取电机转速与轮速差值
        + EKEBA:
          + 前电机减速比：10.79
          + 后电机减速比：10.81
        + EKECA：
          + 前电机减速比：无前电机
          + 后电机减速比：10.81


        Args:
            stage (_type_): _description_

        Returns:
            _type_: _description_
        """
        # # 根据get_first_time方法计算的时间戳，获取对应的电机转速值
        # # 通过前轮轮速换算前电机转速
        # stage_indexed=stage.set_index("timestamps")
        # front_motor_speed = (
        #     (
        #         stage.loc[slip_time, "WhlSpdFL_122"]
        #         + stage.loc[slip_time, "WhlSpdFR_122"]
        #     )
        #     / 2
        #     * 1000
        #     * FRONT_MOTOR_TRANSMISSION_RATIO
        #     / (2 * np.pi * R)
        #     * 60
        # )
        # # 通过后轮轮速换算后电机转速
        # rear_motor_speed = (
        #     (
        #         stage.loc[slip_time, "WhlSpdRL_122"]
        #         + stage.loc[slip_time, "WhlSpdRR_122"]
        #     )
        #     / 2
        #     * 1000
        #     * REAR_MOTOR_TRANSMISSION_RATIO
        #     / (2 * np.pi * R)
        #     * 60
        # )
        front_motor_speed = stage.loc[slip_time, "FMSpd_242"]
        rear_motor_speed = stage.loc[slip_time, "RMSpd_250"]
        # 计算前后电机转速差值
        return front_motor_speed - rear_motor_speed

    def get_single_file_metrics(
        self,
        file_path,
        stage_filters={
            "AccPdlPosn_342": (0, 40),
        },
    ):
        data = pd.read_csv(file_path)
        data["original_index"] = data.index  # 保存原始索引
        slice_idx = self.get_stage_idxs(
            data,
            stage_filters,
        )
        # 单个文件中的数据有多个stage，将多个stage的指标合并为一个DataFrame
        metrics_df = pd.DataFrame()
        stages = self.get_file_stage(data, slice_idx)

        for stage in stages:
            # stage = stage.set_index("timestamps")
            # 计算每个阶段的指标
            stage_metrics = pd.DataFrame()
            # 方向盘转角变化量
            _change_metrics = self.get_start2end_change(stage)
            for signal, value in _change_metrics.items():
                stage_metrics[signal] = [value]
            # 首次滑转时间
            slip_time = self.get_slip_time(stage)
            if slip_time is not None:
                stage_metrics["slip_first_time"] = stage.loc[slip_time, "timestamps"]
                # 首次滑转电机转速差
                stage_metrics["monitor_diff"] = self.get_monitor_diff(stage, slip_time)
                # 首次滑转轮速
                _whl_metrics = self.get_slip_whlspd(stage, slip_time)
                for _whl, whl_spd in _whl_metrics.items():
                    stage_metrics[_whl] = whl_spd
            # 计算stage内平均油门throttle_mean
            throttle_mean = stage["AccPdlPosn_342"].mean()
            # 将列表[20,30,40,50,60,70,80,90,100]中最接近throttle_mean的值，作为此阶段的油门开度
            throttle = min(
                [20, 30, 40, 50, 60, 70, 80, 90, 100],
                key=lambda x: abs(x - throttle_mean),
            )
            stage_metrics["throttle"] = throttle
            metrics_df = pd.concat([metrics_df, stage_metrics], ignore_index=True)
        # 将所有阶段的指标合并为一个DataFrame
        return metrics_df

    def get_all_metrics(self):
        # 将所有文件中的指标合并为一个DataFrame

        for group_name in self.grouped_files:
            grouped_files_metrics = pd.DataFrame()
            for file in self.grouped_files[group_name]:
                single_file_metrics = self.get_single_file_metrics(
                    file,
                    {
                        "AccPdlPosn_342": (0, 40),
                    },
                )
                grouped_files_metrics = pd.concat(
                    [grouped_files_metrics, single_file_metrics], ignore_index=True
                )
            self.all_metrics[group_name] = grouped_files_metrics
        return self



if __name__ == "__main__":

    pass
