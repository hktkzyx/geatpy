# -*- coding: utf-8 -*-
import numpy as np

import geatpy as ea


class IMOP4(ea.Problem):  # 继承Problem父类

    def __init__(self, M=None, Dim=3, Alpha=0.05,
                 K=5):  # M : 目标维数；Dim : 决策变量维数
        name = 'IMOP4'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = 1  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = 0  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = 0  # 决策变量下界
        ub = 1  # 决策变量上界
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
        self.Alpha = Alpha
        self.K = K

    def evalVars(self, Vars):  # 目标函数
        temp = np.abs(np.mean(Vars[:, :self.K],
                              1))**self.Alpha  # 取绝对值，避免因浮点数精度而导致的小于0
        g = np.sum((Vars[:, self.K:] - 0.5)**2, 1)
        ObjV1 = (1 + g) * temp
        ObjV2 = (1 + g) * (temp + np.sin(10 * np.pi * temp) / 10)
        ObjV3 = (1 + g) * (1 - temp)
        f = np.array([ObjV1, ObjV2, ObjV3]).T
        return f

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        Num = 10000  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, Num)
        ObjV2 = ObjV1 + np.sin(10 * np.pi * ObjV1) / 10
        ObjV3 = 1 - ObjV1
        return np.array([ObjV1, ObjV2, ObjV3]).T
