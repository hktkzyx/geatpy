# -*- coding: utf-8 -*-
import pathlib
from typing import Iterable, Union

import numpy as np

import geatpy as ea


class Population:
    """种群类是用来存储种群相关信息的一个类.

    Parameters
    ----------
    Encoding : str or None
        染色体编码方式。
        'BG':二进制/格雷编码；
        'RI':实整数编码，即实数和整数的混合编码；
        'P':排列编码。
        「实值编码」包含实整数编码和排列编码，
        它们共同的特点是染色体不需要解码即可直接表示对应的决策变量。
        「实整数」指的是种群染色体既包含实数的小数，也包含实数的整数。
        特殊用法：
        设置`Encoding=None`，此时种群类的`Field`,`Chrom`成员属性将被设置为`None`，
        种群将不携带与染色体直接相关的信息，可以减少不必要的数据存储，
        这种用法可以在只想统计非染色体直接相关的信息时使用，
        尤其可以在多种群进化优化过程中对个体进行统一的适应度评价时使用。
    Field : array_like or tuple of np.ndarray
        译码矩阵，可以是`FieldD`或`FieldDR`（详见Geatpy数据结构）。
        2.7.0版本之后，可以把问题类对象的`varTypes`、`ranges`、`borders`放到一个元组中传入到此处，
        即`Field = (varTypes, ranges, borders)`
        此时将调用`geatpy.crtfld`自动构建译码矩阵。
    NIND : int, default 0
        种群个体数量
    Chrom : (M,N) array_like, optional
        种群染色体矩阵，每一行对应一个个体的一条染色体。
    ObjV : (M,N) array_like, optional
        种群目标函数值矩阵，每一行对应一个个体的目标函数值，每一列对应一个目标。
    FitnV : (M,1) array_like, optional
        种群个体适应度列向量，每个元素对应一个个体的适应度，最小适应度为0。
    CV : (M,N) array_like, optional
        Constraint Violation Value 是用来定量描述违反约束条件程度的矩阵。
        每行对应一个个体，每列对应一个约束。当没有设置约束条件时，CV设置为None。
    Phen : (M,N) array_like, optional
        种群表现型矩阵。种群各染色体解码后所代表的决策变量所组成的矩阵。

    Attributes
    ----------
    sizes : int
        种群规模，即种群的个体数目。
    ChromNum : int, default 1
        染色体的数目，即每个个体有多少条染色体。

    Notes
    -----
    实例化种群对象，例如：
    ```python
    import geatpy as ea
    population = ea.Population(Encoding, Field, NIND)，
    ```
    `NIND`为所需要的个体数。
    此时得到的`population`还没被真正初始化，仅仅是完成种群对象的实例化。
    该构造函数必须传入`Chrom`，才算是完成种群真正的初始化。
    也可以只传入`Encoding`, `Field`以及`NIND`来完成种群对象的实例化，
    调用`initChrom`方法或者赋值其他属性完成初始化。
    特殊用法1：
    可以利用`ea.Population(Encoding, Field, 0)`来创建一个“空种群”,即不含任何个体的种群对象。
    特殊用法2：
    直接用`ea.Population(Encoding)`构建一个只包含编码信息的空种群。
    """

    def __init__(self,
                 Encoding,
                 Field=None,
                 NIND=0,
                 Chrom=None,
                 ObjV=None,
                 FitnV=None,
                 CV=None,
                 Phen=None):
        """
        """
        if isinstance(NIND, int) and NIND >= 0:
            self.sizes = NIND
        else:
            raise RuntimeError('error in Population: Size error. '
                               '(种群规模设置有误，必须为非负整数。)')
        self.ChromNum = 1
        self.Encoding = Encoding
        if Encoding is None:
            self.Field = None
        elif type(Field) == tuple:
            self.Field = ea.crtfld(Encoding, *Field)
        else:
            self.Field = np.array(Field)
        self.Chrom = np.array(Chrom) if Chrom is not None else None
        self.Lind = self.Chrom.shape[1] if self.Chrom is not None else 0
        self.ObjV = np.array(ObjV) if ObjV is not None else None
        self.FitnV = np.array(FitnV) if FitnV is not None else None
        self.CV = np.array(CV) if CV is not None else None
        self.Phen = np.array(Phen) if Phen is not None else None

    def initChrom(self, NIND=None):
        """初始化种群染色体矩阵.

        Parameters
        ----------
        NIND : int, optional
            用于修改种群规模。
            当其不缺省时，种群在初始化染色体矩阵前会把种群规模调整为`NIND`。
        """
        if NIND is not None and isinstance(NIND, int) and NIND > 0:
            self.sizes = NIND  # 重新设置种群规模
        self.Chrom = ea.crtpc(self.Encoding, self.sizes, self.Field)  # 生成染色体矩阵
        self.Lind = self.Chrom.shape[1]  # 计算染色体的长度
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def decoding(self):
        """种群染色体解码."""
        if self.Encoding == 'BG':  # 此时Field实际上为FieldD
            Phen = ea.bs2ri(self.Chrom, self.Field)  # 把二进制/格雷码转化为实整数
        elif self.Encoding in ['RI', 'P']:
            Phen = self.Chrom.copy()
        else:
            raise RuntimeError(
                "Error in Population.decoding: Encoding must be "
                "'BG' or 'RI' or 'P'. "
                "(编码设置有误，解码时Encoding必须为'BG', 'RI' 或 'P'。)")
        return Phen

    def copy(self):
        """种群的复制.

        假设`pop`是一个种群矩阵，那么：`pop1 = pop.copy()`即可完成对`pop`种群的复制。
        """
        return Population(self.Encoding,
                          self.Field,
                          self.sizes,
                          self.Chrom,
                          self.ObjV,
                          self.FitnV,
                          self.CV,
                          self.Phen)

    def _parse_slice(self, indices: slice) -> np.ndarray:
        """Convert slice to index array.

        Parameters
        ----------
        indices : slice
            A slice instance.

        Returns
        -------
        np.ndarray of int
            Indices.
        """
        if indices.start is None:
            start = 0
        elif -self.sizes <= indices.start < 0:
            start = self.sizes + indices.start
        elif 0 <= indices.start < self.sizes:
            start = indices.start
        else:
            raise IndexError(f'index {indices.start} is out of bounds '
                             f'with size {self.sizes}')
        if indices.stop is None:
            stop = self.sizes
        elif -self.sizes <= indices.stop < 0:
            stop = self.sizes + indices.stop
        elif 0 <= indices.stop < self.sizes:
            stop = indices.stop
        else:
            raise IndexError(f'index {indices.stop} is out of bounds '
                             f'with size {self.sizes}')
        return (np.arange(start, stop, indices.step)
                if indices.step is not None else np.arange(start, stop))

    def __getitem__(self,
                    index: Union[slice, int, Iterable[int], Iterable[bool]]):
        """种群的切片，即根据index下标向量选出种群中相应的个体组成一个新的种群.

        假设pop是一个包含5个个体的种群矩阵，
        那么：
        ```python
        pop[:2]
        pop[[0, 1]]
        pop[[True, True, False, False, False]]
        ```
        均可表示种群里第 1、2 个个体组成的新种群。

        Notes
        -----
        该函数不对传入的index参数的合法性进行更详细的检查。
        """
        if isinstance(index, slice):
            indices = self._parse_slice(index)
        else:
            indices = np.asarray(index).flatten()
        individual_number = (np.count_nonzero(indices)
                             if indices.dtype == bool else indices.size)
        return Population(
            self.Encoding,
            self.Field,
            individual_number,
            self.Chrom[indices] if self.Chrom is not None else None,
            self.ObjV[indices] if self.ObjV is not None else None,
            self.FitnV[indices] if self.FitnV is not None else None,
            self.CV[indices] if self.CV is not None else None,
            self.Phen[indices] if self.Phen is not None else None)

    def shuffle(self):
        """打乱种群个体的个体顺序."""
        shuff = np.arange(self.sizes)
        np.random.shuffle(shuff)  # 打乱顺序
        self.Chrom = self.Chrom[shuff, :] if self.Chrom is not None else None
        self.ObjV = self.ObjV[shuff, :] if self.ObjV is not None else None
        self.FitnV = self.FitnV[shuff] if self.FitnV is not None else None
        self.CV = self.CV[shuff, :] if self.CV is not None else None
        self.Phen = self.Phen[shuff, :] if self.Phen is not None else None

    def _can_set(self, pop) -> bool:
        """Return whether population can be set."""
        if not self._can_add(pop):
            return False
        if ((self.ObjV is not None and pop.ObjV is None)
                or (self.ObjV is None and pop.ObjV is not None)):
            return False
        if ((self.FitnV is not None and pop.FitnV is None)
                or (self.FitnV is None and pop.FitnV is not None)):
            return False
        if ((self.CV is not None and pop.CV is None)
                or (self.CV is None and pop.CV is not None)):
            return False
        if ((self.Phen is not None and pop.Phen is None)
                or (self.Phen is None and pop.Phen is not None)):
            return False
        return True

    def __setitem__(self,
                    index: Union[slice, int, Iterable[int], Iterable[bool]],
                    pop):  # 种群个体赋值（种群个体替换）
        """种群个体的赋值.

        假设pop是一个包含多于2个个体的种群矩阵，pop1是另一个包含2个个体的种群矩阵，
        那么
        ```python
        pop[:2] = pop1
        ```
        即可完成将pop种群的第1、2个个体赋值为pop1种群的个体。

        Notes
        -----
        该函数不对传入的index参数的合法性进行更详细的检查。
        此外，进行种群个体替换后，该函数不会对适应度进行主动重置，
        如果因个体替换而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。
        """
        if isinstance(index, slice):
            indices = self._parse_slice(index)
        else:
            indices = np.asarray(index).flatten()
        if not self._can_set(pop):
            raise RuntimeError('Population not match. (种群不一致。)')
        if self.Chrom is not None:
            self.Chrom[indices] = pop.Chrom
        if self.ObjV is not None:
            self.ObjV[indices] = pop.ObjV
        if self.FitnV is not None:
            self.FitnV[indices] = pop.FitnV
        if self.CV is not None:
            self.CV[indices] = pop.CV
        if self.Phen is not None:
            self.Phen[indices] = pop.Phen

    def _can_add(self, pop) -> bool:
        """Return whether the population instance can be added.

        Parameters
        ----------
        pop : ea.Population
            种群对象.

        Returns
        -------
        bool
            是否可加
        """
        if self.Encoding is None:
            return True
        # 两种群染色体的编码方式必须一致
        if self.Encoding != pop.Encoding:
            return False
        # 种群染色体矩阵未初始化
        if self.Chrom is None or pop.Chrom is None:
            return False
        # 两者的译码矩阵必须一致
        if not np.array_equal(self.Field, pop.Field):
            return False
        return True

    def __add__(self, pop):
        """种群个体合并.

        假设pop1, pop2是两个种群，它们的个体数可以相等也可以不相等，
        此时 pop = pop1 + pop2，即可完成对pop1和pop2两个种群个体的合并。

        Notes
        -----
        进行种群合并后，该函数不会对适应度进行主动重置，
        如果因种群合并而需要重新对所有个体的适应度进行评价，则需要手写代码更新种群的适应度。
        """
        if not self._can_add(pop):
            raise RuntimeError(
                'Error in Population: Population cannot be added.'
                '(种群不可加。)')
        return Population(
            self.Encoding,
            self.Field,
            self.sizes + pop.sizes,
            np.vstack([self.Chrom, pop.Chrom])
            if self.Chrom is not None and pop.Chrom is not None else None,
            np.vstack([self.ObjV, pop.ObjV])
            if self.ObjV is not None and pop.ObjV is not None else None,
            np.vstack([self.FitnV, pop.FitnV])
            if self.FitnV is not None and pop.FitnV is not None else None,
            np.vstack([self.CV, pop.CV])
            if self.CV is not None and pop.CV is not None else None,
            np.vstack([self.Phen, pop.Phen])
            if self.Phen is not None and pop.Phen is not None else None)

    def __len__(self):
        """计算种群规模.

        假设pop是一个种群，那么len(pop)即可得到该种群的个体数。
        实际上，种群规模也可以通过pop.sizes得到。
        """
        return self.sizes

    def save(self, dirName='Population Info'):
        """保存种群信息到`dirName`文件夹下.

        "Encoding.txt"保存种群的染色体编码；
        "Field.csv"保存种群染色体的译码矩阵；
        "Chrom.csv"保存种群的染色体矩阵；
        "ObjV.csv"保存种群的目标函数矩阵；
        "FitnV.csv"保存种群个体的适应度列向量；
        "CV.csv"保存种群个体的违反约束程度矩阵；
        "Phen.csv"保存种群染色体表现型矩阵；
        注意：该函数不会对种群的合法性进行检查。

        Parameters
        ----------
        dirName : path_like
            文件夹
        """
        folder = pathlib.Path(dirName)
        if not folder.exists():
            folder.mkdir()
        with open(folder / 'Encoding.txt', 'w') as file:
            file.write(self.Encoding)
        if self.Field is not None:
            np.savetxt(folder / 'Field.csv', self.Field, delimiter=',')
        if self.Chrom is not None:
            np.savetxt(folder / 'Chrom.csv', self.Chrom, delimiter=',')
        if self.ObjV is not None:
            np.savetxt(folder / 'ObjV.csv', self.ObjV, delimiter=',')
        if self.FitnV is not None:
            np.savetxt(folder / 'FitnV.csv', self.FitnV, delimiter=',')
        if self.CV is not None:
            np.savetxt(folder / 'CV.csv', self.CV, delimiter=',')
        if self.Phen is not None:
            np.savetxt(folder / 'Phen.csv', self.Phen, delimiter=',')

    def getInfo(self):
        """获取种群的设置信息."""
        return {
            'Type': 'Population',
            'Population Encoding:': self.Encoding,
            'Population ChromNum': self.ChromNum,
            'Population Field:': self.Field,
            'Population size:': self.sizes
        }

    def __str__(self):
        info = self.getInfo()
        info['Population Chrom'] = self.Chrom
        info['Population Lind'] = self.Lind
        info['Population FitnV'] = self.FitnV
        info['Population ObjV'] = self.ObjV
        info['Population CV'] = self.CV
        info['Population Phen'] = self.Phen
        return str(info)
