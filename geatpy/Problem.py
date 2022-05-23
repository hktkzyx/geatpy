# -*- coding: utf-8 -*-
import pathlib
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np

import geatpy as ea


class Problem:
    """存储与待求解问题相关信息的类.

    Parameters
    ----------
    name : str
        问题名称（可以自由设置名称）。
    M : int
        目标维数，即有多少个优化目标。
    maxormins : array_like of int
        目标函数最小最大化标记，1表示最小化，-1表示最大化.
        例如：[1,1,-1,-1]，表示前2个目标是最小化，后2个目标是最大化。
    Dim : int
        决策变量维数，即有多少个决策变量。
    varTypes : array_like of int
        连续或离散标记. `0`表示对应的决策变量是连续的；`1`表示对应的变量是离散的。
    lb : array_like of float
        存储着各个变量的下界。
    ub : array_like of float
        存储着各个变量的上界。
    lbin : array_like of int, default `1`
        是否包含下边界。`0`表示不含，即开边界，`1`表示包含，即闭边界。
    ubin : array_like of int, default `1`
        是否包含上边界。`0`表示不含，即开边界，`1`表示包含，即闭边界。
    aimFunc : function with respect to `Population`, optional
        目标函数，需要的参数是`Population`的类对象。
        如果是`None`，则使用类中定义的`aimFunc`方法。
    evalVars : function, optional
        用于直接传入决策变量矩阵来计算对应的目标函数矩阵和违反约束程度矩阵。
        如果是`None`，则使用类中定义的`evalVars`方法。
    calReferObjV : function, optional
        计算目标函数参考值。
        如果是`None`，则使用类中定义的`calReferObjV`方法。
    ReferObjV_path : path_like, optional
        目标函数参考值存储位置。
        如果是`None`，则默认为`referenceObjV`文件夹下的「问题名称_目标维数_决策变量个数.csv」CSV 文件.

    Attributes
    ----------
    ranges : (2,N) np.ndarray
        决策变量范围矩阵，第一行对应决策变量的下界，第二行对应决策变量的上界。
    borders : (2,N) np.ndarray
        决策变量范围的边界矩阵，第一行对应决策变量的下边界，第二行对应决策变量的上边界.
        `0`表示范围中不含边界，`1`表示范围包含边界。
    ReferObjV : (M,N) np.ndarray
        存储着目标函数参考值的矩阵，每一行对应一组目标函数参考值，每一列对应一个目标函数。
        即`M`组目标函数参考值，每组`N`个目标函数值.
    TinyReferObjV : (M,N) np.ndarray
        从ReferObjV中均匀抽取的数目更少的目标函数参考值矩阵。
    函数:
        aimFunc(pop)      : 目标函数，需要在继承类即自定义的问题类中实现，或是传入已实现的函数。
                            其中pop为Population类的对象，代表一个种群，
                            pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
                            该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
                            若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
                            该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中。
                            例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
                            此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。
                            注意：在子类中，aimFunc()和evalVars()两者只能重写一个。

        evalVars(v)       : 用于直接传入决策变量矩阵来计算对应的目标函数矩阵和违反约束程度矩阵。该函数需要被子类重写。

        evaluation(pop)   : 调用aimFunc()或evalVars()计算传入种群的目标函数值和违反约束程度。

        calReferObjV()    : 计算目标函数参考值，需要在继承类中实现，或是传入已实现的函数。

        getReferObjV()    : 获取目标函数参考值。
    """

    def __init__(self,
                 name: str,
                 M: int,
                 maxormins: Union[int, Iterable[int]],
                 Dim: int,
                 varTypes: Union[int, Iterable[int]],
                 lb: Union[float, Iterable[float]],
                 ub: Union[float, Iterable[float]],
                 lbin: Union[int, Iterable[int]] = 1,
                 ubin: Union[int, Iterable[int]] = 1,
                 aimFunc: Optional[Callable[[ea.Population], None]] = None,
                 evalVars: Optional[Callable] = None,
                 calReferObjV: Optional[Callable] = None,
                 ReferObjV_path=None):
        self.name = name
        self.M = M
        self.maxormins = np.broadcast_to(maxormins, M)
        self.Dim = Dim
        self.varTypes = np.broadcast_to(varTypes, Dim)
        self.lb = np.broadcast_to(lb, Dim)
        self.ub = np.broadcast_to(ub, Dim)
        self.ranges = np.vstack((self.lb, self.ub))  # 初始化ranges（决策变量范围矩阵）
        self.borders = np.vstack(
            (np.broadcast_to(lbin, Dim),
             np.broadcast_to(ubin, Dim)))  # 初始化borders（决策变量范围边界矩阵）
        self.aimFunc = aimFunc if aimFunc is not None else self.aimFunc  # 初始化目标函数接口
        self.evalVars = evalVars if evalVars is not None else self.evalVars
        self.calReferObjV = calReferObjV if calReferObjV is not None else self.calReferObjV  # 初始化理论最优值计算函数接口
        self.ReferObjV = self.getReferObjV(
            filepath=ReferObjV_path)  # 计算目标函数参考值
        if self.ReferObjV is not None:
            self.TinyReferObjV = _extract_ReferObjV_with_limit_num(
                self.ReferObjV, 100)
        else:
            self.TinyReferObjV = None

    def aimFunc(self, pop: ea.Population) -> None:
        """用于计算整个种群的目标函数矩阵和违反约束程度矩阵的目标函数.

        其中pop为Population类的对象，代表一个种群，
        pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵，
        该函数根据该Phen计算得到种群所有个体的目标函数值组成的矩阵，并将其赋值给pop对象的ObjV属性。
        若有约束条件，则在计算违反约束程度矩阵CV后赋值给pop对象的CV属性（详见Geatpy数据结构）。
        该函数不返回任何的返回值，求得的目标函数值保存在种群对象的ObjV属性中。
        例如：population为一个种群对象，则调用aimFunc(population)即可完成目标函数值的计算，
        此时可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。

        Parameters
        ----------
        pop : ea.Population
            种群对象.

        Raises
        ------
        RuntimeError
            没有重写，默认抛出异常.
        """
        raise RuntimeError(
            'Error in Problem: aim function has not been initialized. '
            '(未在问题子类中设置目标函数！)')

    def evalVars(self,
                 Vars) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Evaluate variables 用于直接传入决策变量矩阵来计算对应的目标函数矩阵和违反约束程度矩阵.

        有约束时
        ```python
        ObjV, CV = evalVars(Vars)
        ```
        没有约束时
        ```python
        ObjV = evalVars(Vars)
        ```

        Parameters
        ----------
        Vars : (M,) array_like
            决策变量。每一行代表一组决策变量。

        Returns
        -------
        ObjV : (M,N) np.ndarray
            目标函数值。`M`个决策变量，`N`个目标函数值。
        CV : (M,) np.ndarray
            违反约束程度矩阵。如果没有约束，则不返回这一项。

        Raises
        ------
        RuntimeError
            没有重写，默认抛出异常.
        """
        raise RuntimeError(
            'Error in Problem: aim function has not been initialized. '
            '(未在问题子类中设置目标函数！)')

    @staticmethod
    def single(func):
        """单组决策变量装饰器.

        装饰器single。通过给目标函数添加装饰器，可以更专注于问题的模型本身。因为此时传入自定义目标函数的只有一组决策变量。
        装饰后的函数可以作为参数传入`evalVars`。详见demo.
        """

        def wrapper(param):
            param = np.asarray(param)
            param = np.atleast_2d(param)
            if param.ndim > 2:
                raise TypeError(
                    'Invalid input parameter of evalVars. '
                    'evalVars can be convert to a 2D numpy.ndarray. '
                    '(evalVars的传入参数非法，必须可以转换为 2D numpy.ndarray。)')
            ObjV = []
            CV = []
            for indiv in param:
                return_object = func(indiv)
                if type(return_object) != tuple:
                    ObjV.append(np.atleast_1d(return_object))
                else:
                    obj_i, cv_i = return_object
                    ObjV.append(np.atleast_1d(obj_i))
                    CV.append(np.atleast_1d(cv_i))
            return (np.vstack(ObjV), np.vstack(CV)) if CV else np.vstack(ObjV)

        return wrapper

    def evaluation(self, pop: ea.Population):
        """调用aimFunc()或evalVars()计算传入种群的目标函数值和违反约束程度.

        优先调用`aimFunc`，如果`aimFunc`没有重写，则调用`evalVars`。

        Parameters
        ----------
        pop : ea.Pupulation
            种群对象.

        Raises
        ------
        RuntimeError
            `aimFunc`和`evalVars`都未重写.
        """
        # 先尝试执行aimFunc()
        try:
            self.aimFunc(pop)
            return
        except RuntimeError:
            pass
        try:
            return_object = self.evalVars(pop.Phen)
            if type(return_object) != tuple:
                pop.ObjV = return_object
            else:
                pop.ObjV, pop.CV = return_object
        except RuntimeError:
            raise RuntimeError(
                'Error in Problem: '
                'one of the function aimFunc and evalVars should be rewritten. '
                '(aimFunc和evalVars两个函数必须至少有一个被子类重写。)')

    def calReferObjV(self) -> Optional[np.ndarray]:
        """计算目标函数的参考值.

        如果待优化的模型知道理论全局最优解，则可以在自定义问题类里重写该函数，求出理论全局最优解对应的目标函数值矩阵。

        Returns
        -------
        (M,N) np.ndarray or None
            目标函数参考值。每一行代表一组参考值，每一列代表一个目标函数值。
            如果是`None`，则表示没有参考值。
        """
        return None

    def getReferObjV(self, reCalculate: bool = False,
                     filepath=None) -> np.ndarray:
        """获取目标函数参考值.

        该函数用于读取/计算问题的目标函数参考值。
        这个参考值可以是理论上的全局最优解的目标函数值，也可以是人为设定的非最优的目标函数参考值。
        在获取或计算出理论全局最优解后，结果将保存到`filepath`文件内。

        Parameters
        ----------
        reCalculate : bool, default `False`
            表示是否要调用`calReferObjV`来重新计算目标函数参考值。
        filepath : path_like or None
            目标函数参考值文件。
            当缺省时结果默认保存到`referenceObjV`文件夹下的「问题名称_目标维数_决策变量个数.csv」CSV 文件。

        Returns
        -------
        refer_obj_value : (M,N) np.ndarray
            存储着目标函数参考值的矩阵，每一行对应一组目标函数参考值，每一列对应一个目标函数。
        """
        if filepath is None:
            filepath = pathlib.Path('referenceObjV',
                                    f'{self.name}_M{self.M}_D{self.Dim}.csv')
        else:
            filepath = pathlib.Path(filepath)
        if not reCalculate and filepath.exists():
            return _read_ReferObjV_from_csv(filepath)
        # 若找不到数据，则调用calReferObjV()计算目标函数参考值
        refer_obj_value = self.calReferObjV()
        if refer_obj_value is not None:
            refer_obj_value = np.atleast_2d(refer_obj_value)
            # 简单检查referenceObjV的合法性
            if refer_obj_value.ndim > 2 or refer_obj_value.shape[1] != self.M:
                raise RuntimeError(
                    'Error: ReferenceObjV is illegal. (目标函数参考值矩阵的数据格式不合法，请检查自定义问题类中的calReferObjV('
                    ')函数的代码，假如没有目标函数参考值，则在问题类中不需要定义calReferObjV()函数。)')
            # 保存数据
            _write_ReferObjV_to_csv(refer_obj_value, filepath)
        return refer_obj_value

    def __str__(self):
        info = {
            'name': self.name,
            'M': self.M,
            'maxormins': self.maxormins,
            'Dim': self.Dim,
            'varTypes': self.varTypes,
            'lb': self.lb,
            'ub': self.ub,
            'borders': self.borders
        }
        return str(info)


def _read_ReferObjV_from_csv(csvpath) -> np.ndarray:
    """Read reference objective values from the CSV file.

    Parameters
    ----------
    csvpath : path_like
        保存目标函数参考值的 CSV 文件。

    Returns
    -------
    ReferObjV : (M,N) np.ndarray
        目标函数参考值。每一行对应一组目标函数参考值，每一列对应一个目标函数。
    """
    return np.loadtxt(csvpath, delimiter=',', ndmin=2)


def _write_ReferObjV_to_csv(ReferObjV, csvpath) -> None:
    """Write reference objective values to a CSV file.

    Parameters
    ----------
    ReferObjV : matched (M,N) array_like
        目标函数参考值。
    csvpath : path_like
        保存目标函数参考值的 CSV 文件。
    """
    refer_objv = np.asarray(ReferObjV)
    csvpath = pathlib.Path(csvpath)
    if not csvpath.resolve().parent.exists():
        csvpath.resolve().parent.mkdir()
    np.savetxt(csvpath, refer_objv, delimiter=',')


def _extract_ReferObjV_with_limit_num(ReferObjV, max_num) -> np.ndarray:
    """Return part of reference objective values if exceed the maximum number.

    Parameters
    ----------
    ReferObjV : matched (M,N) array_like
        目标函数参考值。
    max_num : int
        最大数目。
    """
    refer_obj_value = np.asarray(ReferObjV)
    if refer_obj_value.shape[0] > max_num:
        indices = np.linspace(0, refer_obj_value.shape[0] - 1,
                              num=max_num).astype(np.int32)
        return refer_obj_value[indices, :]
    return refer_obj_value
