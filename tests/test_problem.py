import numpy as np
import pytest

import geatpy


@geatpy.Problem.single
def eval_function(indiv):
    return np.sum(indiv)


@geatpy.Problem.single
def eval_function_with_cv(indiv):
    return np.sum(indiv), np.asarray(indiv)


def test_Problem_single():
    indivs = [[1, 2, 3], [4, 5, 6]]
    assert np.array_equal(eval_function(indivs), [[6], [15]])
    obj, cv = eval_function_with_cv(indivs)
    assert np.array_equal(obj, [[6], [15]])
    assert np.array_equal(cv, [[1, 2, 3], [4, 5, 6]])


def test_Problem_single_with_single_indiv():
    indivs = [1, 2, 3]
    assert np.array_equal(eval_function(indivs), [[6]])
    obj, cv = eval_function_with_cv(indivs)
    assert np.array_equal(obj, [[6]])
    assert np.array_equal(cv, [[1, 2, 3]])


@pytest.fixture
def population_only_with_Phen():
    indivs = np.array([[1, 2, 3], [4, 5, 6]])
    yield geatpy.Population(None, Phen=indivs)


def test_Problem_evaluation_aimFunc_evalVars_invalid(
        population_only_with_Phen):
    pop = population_only_with_Phen
    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3))
    with pytest.raises(RuntimeError):
        problem.evaluation(pop)


def test_Problem_evaluation_with_evalVars(population_only_with_Phen):
    pop = population_only_with_Phen
    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3),
                             evalVars=eval_function)
    problem.evaluation(pop)
    assert np.array_equal(pop.ObjV, [[6], [15]])


def test_Problem_evaluation_with_aimFunc_and_evalVars(
        population_only_with_Phen):

    def aim_func_example(pop):
        pop.ObjV = [[10]]

    pop = population_only_with_Phen
    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3),
                             aimFunc=aim_func_example,
                             evalVars=eval_function)
    problem.evaluation(pop)
    assert np.array_equal(pop.ObjV, [[10]])


@pytest.fixture
def problem_without_calReferObjV():
    yield geatpy.Problem('test',
                         M=1,
                         maxormins=[1],
                         Dim=3,
                         varTypes=np.zeros(3),
                         lb=np.zeros(3),
                         ub=10 * np.ones(3),
                         evalVars=eval_function)


def test_Problem_getReferObjV_from_file(tmp_path,
                                        problem_without_calReferObjV):
    problem = problem_without_calReferObjV
    csvpath = tmp_path / 'refer_obj_value.csv'
    expect = np.arange(6).reshape((2, 3))
    np.savetxt(csvpath, expect, delimiter=',')
    assert np.array_equal(problem.getReferObjV(False, csvpath), expect)


def test_Problem_getReferObjV_with_valid_calReferObjV(
        tmp_path, problem_without_calReferObjV):

    csvpath = tmp_path / 'reference_obj_value' / 'refer_obj_value.csv'

    def valid_cal_refer_obj_value():
        return np.arange(10)[:, np.newaxis]

    problem = problem_without_calReferObjV
    problem.calReferObjV = valid_cal_refer_obj_value
    result = problem.getReferObjV(filepath=csvpath)
    result_in_file = np.loadtxt(csvpath, delimiter=',')
    expect = np.arange(10)[:, np.newaxis]
    np.array_equal(result, expect)
    np.array_equal(result_in_file, expect)


def test_Problem_getReferObjV_with_invalid_calReferObjV(
        tmp_path, problem_without_calReferObjV):

    csvpath = tmp_path / 'reference_obj_value' / 'refer_obj_value.csv'

    def invalid_cal_refer_obj_value():
        return np.arange(10).reshape((2, 5))

    problem = problem_without_calReferObjV
    problem.calReferObjV = invalid_cal_refer_obj_value
    with pytest.raises(RuntimeError):
        problem.getReferObjV(filepath=csvpath)


def test_Problem_getReferObjV_without_calReferObjV(
        tmp_path, problem_without_calReferObjV):
    csvpath = tmp_path / 'reference_obj_value' / 'refer_obj_value.csv'
    problem = problem_without_calReferObjV
    assert problem.getReferObjV(False, csvpath) is None
    assert not csvpath.exists()


def test_Problem_with_short_TinyReferObjV(tmp_path):
    csvpath = tmp_path / 'reference_obj_value' / 'refer_obj_value.csv'

    def short_cal_refer_obj_value():
        return np.arange(10)[:, np.newaxis]

    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3),
                             evalVars=eval_function,
                             calReferObjV=short_cal_refer_obj_value,
                             ReferObjV_path=csvpath)
    expect = np.arange(10)[:, np.newaxis]
    assert np.array_equal(problem.TinyReferObjV, expect)


def test_Problem_with_long_TinyReferObjV(tmp_path):
    csvpath = tmp_path / 'reference_obj_value' / 'refer_obj_value.csv'

    def long_cal_refer_obj_value():
        return np.ones((200, 1))

    problem = geatpy.Problem('test',
                             M=1,
                             maxormins=[1],
                             Dim=3,
                             varTypes=np.zeros(3),
                             lb=np.zeros(3),
                             ub=10 * np.ones(3),
                             evalVars=eval_function,
                             calReferObjV=long_cal_refer_obj_value,
                             ReferObjV_path=csvpath)
    expect = np.ones((100, 1))
    assert np.array_equal(problem.TinyReferObjV, expect)
