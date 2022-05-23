# -*- coding: utf-8 -*-
import numpy as np

import geatpy as ea


class Griewangk(ea.Problem):  # 继承Problem父类

    def __init__(self, Dim=30):  # Dim : 决策变量维数
        name = 'Griewangk'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = 1  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = 0  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        # 初始化varTypes（决策变量的类型）
        lb = -600  # 决策变量下界
        ub = 600  # 决策变量上界
        lbin = 1  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = 1  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):  # 目标函数
        Nind = Vars.shape[0]
        number = np.tile(np.arange(1, self.Dim + 1), (Nind, 1))
        f = np.array([
            np.sum(((Vars**2) / 4000).T, 0)
            - np.prod(np.cos(Vars / np.sqrt(number)).T, 0) + 1
        ]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）
        referenceObjV = np.array([[0]])
        return referenceObjV
