#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
author: fenwickxie
date: 2025-04-27 11:16:11
filename: setup.py
version: 1.0
"""

if __name__ == "__main__":
    from setuptools import setup

    setup(
        name="data_analysis",
        version="v0.0.2",
        packages=['data','chart_gen','doc_gen'],
        install_requires=[
            "pandas",
            "asammdf",
            "cantools",
            "tqdm",
            "matplotlib",
            "numpy",
        ],
        author="fenwickxie",
        author_email="fenwickxie@outlook.com",
        description="A PyTorch implementation of Single Shot MultiBox Detector",
    )
