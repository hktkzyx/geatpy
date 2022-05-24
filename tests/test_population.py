import numpy as np
import pytest

import geatpy


@pytest.fixture
def population_with_field():
    var_num = 3
    var_types = np.zeros(var_num)
    ranges = np.vstack((np.zeros(var_num), np.ones(var_num)))
    borders = np.ones((2, var_num))
    individual_num = 5
    yield geatpy.Population('BG', (var_types, ranges, borders), individual_num)


@pytest.fixture
def population_without_encoding():
    yield geatpy.Population(None)


def test_population_initChrom(population_with_field):
    pop = population_with_field
    pop.initChrom()
    assert isinstance(pop.Chrom, np.ndarray)
    assert pop.Chrom.ndim == 2
    assert pop.Chrom.shape[0] == 5


def test_population_decoding(population_with_field):
    pop = population_with_field
    pop.initChrom()
    phen = pop.decoding()
    assert isinstance(phen, np.ndarray)
    assert phen.shape == (5, 3)


def test_population_copy(population_with_field):
    pop = population_with_field
    pop_not_init = pop.copy()
    pop.initChrom()
    pop_init = pop.copy()
    assert pop_not_init.Chrom is None
    assert np.array_equal(pop.Chrom, pop_init.Chrom)


def test_population_getitem(population_with_field):
    pop = population_with_field
    pop.initChrom()
    pop.ObjV = np.arange(10).reshape((5, 2))
    selected = pop[:2]
    assert pop.Encoding == selected.Encoding
    assert np.array_equal(pop.Field, selected.Field)
    assert len(selected) == 2
    assert np.array_equal(pop.Chrom[:2, :], selected.Chrom)
    assert np.array_equal(selected.ObjV, [[0, 1], [2, 3]])
    assert selected.FitnV is None
    assert selected.CV is None
    assert selected.Phen is None


def test_population_getitem_with_minus_index(population_with_field):
    pop = population_with_field
    pop.initChrom()
    pop.ObjV = np.arange(10).reshape((5, 2))
    selected = pop[-2:]
    assert pop.Encoding == selected.Encoding
    assert np.array_equal(pop.Field, selected.Field)
    assert len(selected) == 2
    assert np.array_equal(pop.Chrom[-2:, :], selected.Chrom)
    assert np.array_equal(selected.ObjV, [[6, 7], [8, 9]])
    assert selected.FitnV is None
    assert selected.CV is None
    assert selected.Phen is None


def test_population_shuffle(population_with_field):
    pop = population_with_field
    pop.initChrom()
    pop.ObjV = np.arange(10).reshape((5, 2))
    chrom_before_shuffle = pop.Chrom
    pop.shuffle()
    index = np.argwhere(pop.ObjV == 0)
    assert not np.array_equal(chrom_before_shuffle, pop.Chrom)
    assert np.array_equal(chrom_before_shuffle[0], pop.Chrom[index[0, 0]])


def test_population_setitem(population_with_field):
    pop = population_with_field
    with pytest.raises(RuntimeError):
        pop[2:4] = pop[:2]
    pop.initChrom()
    pop.ObjV = np.arange(10).reshape((5, 2))
    pop[2:4] = pop[:2]
    assert np.array_equal(pop.Chrom[2:4], pop.Chrom[:2])
    assert np.array_equal(pop.ObjV[2:4], [[0, 1], [2, 3]])


def test_population_add(population_with_field):
    pop = population_with_field
    with pytest.raises(RuntimeError):
        pop + pop
    pop.initChrom()
    pop.ObjV = np.arange(10).reshape((5, 2))
    sum_pop = pop + pop
    assert len(sum_pop) == 10
    assert np.array_equal(sum_pop.Chrom[:5], sum_pop.Chrom[5:])
    assert np.array_equal(sum_pop.ObjV, np.vstack((pop.ObjV, pop.ObjV)))


def test_population_save(population_with_field, tmp_path):
    pop = population_with_field
    pop.initChrom()
    pop.save(tmp_path)
    encoding_file = tmp_path / 'Encoding.txt'
    field_file = tmp_path / 'Field.csv'
    chrom_file = tmp_path / 'Chrom.csv'
    objv_file = tmp_path / 'ObjV.csv'
    fitnv_file = tmp_path / 'FitnV.csv'
    cv_file = tmp_path / 'CV.csv'
    phen_file = tmp_path / 'Phen.csv'
    assert encoding_file.exists()
    assert field_file.exists()
    assert chrom_file.exists()
    assert not objv_file.exists()
    assert not fitnv_file.exists()
    assert not cv_file.exists()
    assert not phen_file.exists()


def test_add_population_without_encoding(population_with_field,
                                         population_without_encoding):
    with pytest.raises(RuntimeError):
        population_with_field + population_without_encoding
    pop_normal = population_with_field
    pop_normal.initChrom()
    pop_normal.ObjV = np.arange(10).reshape((5, 2))
    pop_special = population_without_encoding
    pop_special.sizes = 1
    pop_special.ObjV = np.array([[5, 5]])
    pop = pop_special + pop_normal
    assert pop.Chrom is None
    assert len(pop) == 6
    assert np.array_equal(pop.ObjV,
                          np.vstack((pop_special.ObjV, pop_normal.ObjV)))


def test_population_getInfo(population_with_field):
    pop = population_with_field
    pop.initChrom()
    pop.ObjV = np.arange(10).reshape((5, 2))
    info = pop.getInfo()
    assert info['Type'] == 'Population'
    assert info['Population Encoding'] == 'BG'
    assert info['Population ChromNum'] == 1
    assert info['Population size'] == 5


def test_population_str(population_without_encoding):
    expect = ("{'Type': 'Population', "
              "'Population Encoding': None, "
              "'Population ChromNum': 1, "
              "'Population Field': None, "
              "'Population size': 0, "
              "'Population Chrom': None, "
              "'Population Lind': 0, "
              "'Population FitnV': None, "
              "'Population ObjV': None, "
              "'Population CV': None, "
              "'Population Phen': None}")
    assert expect == str(population_without_encoding)
