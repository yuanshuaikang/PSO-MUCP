import math
import time
from math import sqrt

f = open(r"D:\PSO-MUCP 代码\IDEA co-location\数据文档\California_POI 13f.csv", "r", encoding="UTF-8")
Instance = []
for line in f:
    temp_2 = []
    temp = line.strip().split(",")
    if temp != [''] and temp != ['Feature', 'Instance', 'LocX', 'LocY', 'Checkin']:
        s = temp[0] + temp[1]
        temp_2.append(s)
        temp_2.append(float(temp[2]))
        temp_2.append(float(temp[3]))
        Instance.append(temp_2)
f.close()
Utility = {'A': 2, 'B': 4, 'C': 8, 'D': 4, "E": 1, "F": 4, "G": 5, "H": 3, "I": 9, "J": 3, 'K': 5, 'L': 10,'M': 8}
Min_utility = 0.1
D = 1300 # 距离阈值越大生成得邻近关系越多
"====================================================================================================================="


start_time = time.time()


# 计算出每个特征的总效用然后将特征按照总效用从大到小排列
def Compute_utility(utility, instance):
    all_utility = 0
    for item in instance:
        all_utility += utility[item[0][0]]
    return all_utility


"=============================================================================================================="


# 计算出所有的实例邻近关系
def computed_neigh(instance_1, instance_2, d):
    x = instance_1[1] - instance_2[1]
    y = instance_1[2] - instance_2[2]
    distance_12 = sqrt(x ** 2 + y ** 2)
    if distance_12 <= d:
        return True
    else:
        return False


def grid_method(i, d):
    """
    采用网格法计算每一个实例的邻近结点，每一个实例只用检查本身所在网格以及与它相邻的八个方向的网格
    :param i: 实例集
    :param d: 网格变成（距离阈值）
    :return: Ns，PNs，SNs
    """
    # 找到整个网格的范围，对网格进行划分，并将实例分配进对应的网格
    x_min = min(x[1] for x in i)
    y_min = min(x[2] for x in i)
    hash_grid = {}
    for elem in i:
        x_order = math.ceil((elem[1] - x_min) / d)
        y_order = math.ceil((elem[2] - y_min) / d)
        if x_order == 0:
            x_order += 1
        if y_order == 0:
            y_order += 1
        hash_grid.setdefault((x_order, y_order), []).append(elem)

    # 根据划分的网格计算每个实例的邻近关系
    Ns = {}
    computed_neigh = lambda x, y, d: (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2 <= d ** 2
    for elem_hash, grid_instances in hash_grid.items():
        for elem_list in grid_instances:
            Ns.setdefault(elem_list[0], [])
            for x in grid_instances:
                if x[0][0] != elem_list[0][0] and computed_neigh(x, elem_list, d):
                    Ns[elem_list[0]].append(x[0])

            # 计算相邻的八个方向
            for delta_x in range(-1, 2):
                for delta_y in range(-1, 2):
                    if delta_x == delta_y == 0:
                        continue
                    adjacent_grid = (elem_hash[0] + delta_x, elem_hash[1] + delta_y)
                    if adjacent_grid in hash_grid:
                        for elem_xy in hash_grid[adjacent_grid]:
                            if elem_xy[0][0] != elem_list[0][0] and computed_neigh(elem_xy, elem_list, d):
                                Ns[elem_list[0]].append(elem_xy[0])
    return Ns


"=============================================================================================================="


# 从k-1阶模式生成k阶候选同位模式，并按照字典序排列
def gen_candidate(c_k_1, k):
    c_k = []  # 存储k阶候选同位模式
    for i in range(len(c_k_1)):  # 因为候选模式集是按照字典序进行排列所以可以按照序号遍历
        for j in range(i + 1, len(c_k_1)):
            if c_k_1[i][: len(c_k_1[i]) - 1] == c_k_1[j][:len(c_k_1[j]) - 1]:
                if k == 2:
                    s = c_k_1[i][len(c_k_1[i]) - 1] + c_k_1[j][len(c_k_1[j]) - 1]
                else:
                    s = c_k_1[i][: len(c_k_1[i]) - 1] + c_k_1[i][len(c_k_1[i]) - 1] + c_k_1[j][len(c_k_1[j]) - 1]
                c_k.append(s)
    return c_k


"=============================================================================================================="


# 生成模式的行实例
def gener_row(nis, can_pattern):
    fake_row_instance = []
    for keys in nis:
        if keys[0] == can_pattern[0]:
            flag = 0
            for f in can_pattern[1: len(can_pattern)]:
                g = 0
                for elem in nis[keys]:
                    # print(elem[0], f, "22222222222")
                    if elem[0] == f:
                        g = 1
                        break
                if g == 0:  # g为0说明没找到相同的特征，这个邻近关系不能生成行实例
                    flag = 1
                    break
            if flag == 0:  # 如果flag=0说明这个邻近关系里包含所有的候选特征
                temp = [[keys]]
                for fu in can_pattern[1: len(can_pattern)]:
                    temp_1 = []
                    for item in nis[keys]:
                        if item[0] == fu:
                            for can_list in temp:
                                kong = can_list.copy()
                                kong.append(item)
                                temp_1.append(kong)
                    temp = temp_1
                for elem in temp:
                    fake_row_instance.append(elem)

    #  验证是否是真的行实例
    true_row_instance = []
    # print(can_pattern, fake_row_instance, "假")
    for elem in fake_row_instance:
        flag = 0
        # print(elem)
        for i in range(1, len(elem) - 1):
            if not all(element in nis[elem[i]] for element in elem[i + 1: len(elem)]):
                flag = 1
        if flag == 0:
            true_row_instance.append(elem)
    # print(true_row_instance, "真")
    return true_row_instance


"=============================================================================================================="


# 计算模式的效用值和效用率
def compute_utility(can_pattern, row, utility, all_utility):
    hash_instance = {}
    for fu in can_pattern:
        hash_instance[fu] = []
    for elem_list in row:
        for elem in elem_list:
            if elem not in hash_instance[elem[0]]:
                hash_instance[elem[0]].append(elem)
    lc = 0
    for keys in hash_instance:
        lc += utility[keys] * len(hash_instance[keys])
    # if can_pattern == 'EFGHJ':
    #     print(lc)
    #     print(hash_instance)
    return lc / all_utility


"=============================================================================================================="


# 得出不包含当前候选模式的剩余特征
def rest_f(can_pattern, utility):
    f_list = utility.keys()
    s = ""
    for f in f_list:
        if f not in can_pattern:
            s += f
    return s


"=============================================================================================================="


# 计算扩展效用
def compute_extend_utility(num, pattern, f, nis, utility, all_utility):
    sub_patter = []
    h = pattern + f
    for sx in h:
        sub_patter.append(h.replace(sx, ""))
    sort_sub = sorted(sub_patter)

    f_instance = []
    i = 1
    for sub in sort_sub:
        if f in sub:
            # 找到行实例
            rows = gener_row(nis, sub)
            for item_list in rows:
                for item in item_list:
                    if item[0] == f and item not in f_instance:
                        f_instance.append(item)
        if i > num:
            break
        i += 1
    vs = (len(f_instance) * utility[f]) / all_utility
    return vs


"=============================================================================================================="


# 计算一阶模式的效用率
def computed_one(F, utility, nis, all_utility):
    f_utility = 0
    for elem in nis:
        if elem[0] == F:
            f_utility += utility[F]
    return f_utility / all_utility


"=============================================================================================================="

# 替代方案示例 - 手动检查内存
import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def my_func():

    ALL_utility = Compute_utility(Utility, Instance)  # U(s)
    # print(ALL_utility)

    pattern_number = 0
    pruning = 0
    k = 2
    High_utility = {}
    C_k_1 = []
    NIS = grid_method(Instance, D)
    print(NIS)
    for one in Utility.keys():
        one_utility = computed_one(one, Utility, NIS, ALL_utility)
        if one_utility >= Min_utility:
            High_utility[one] = one_utility
    no_hope = []
    # print(High_utility)
    while k <= len(Utility.keys()):
        # 生成k阶候选模式
        if k == 2:
            C_k = gen_candidate(sorted(Utility.keys()), k)
        else:
            C_k = gen_candidate(C_k_1, k)
        print(C_k)
        pattern_number += len(C_k)
        for Can_pattern in C_k:
            if len(no_hope) != 0: # 只要有一个无希望模式在候选模式中就跳过这个候选模式
                Flag = 0
                for elem in no_hope:
                    if elem == Can_pattern:
                        Flag = 1
                        continue

                if Flag == 1:
                    continue

            Tc = gener_row(NIS, Can_pattern)
            if len(Tc) == 0:
                continue
            Uc = compute_utility(Can_pattern, Tc, Utility, ALL_utility)
            if Uc >= Min_utility:
                # print(High_utility)
                High_utility[Can_pattern] = Uc
            else:
                # print(Can_pattern, Uc)
                f_c = rest_f(Can_pattern, Utility)
                vss = 0
                for r_f in f_c:
                    vss += compute_extend_utility(k - 2, Can_pattern, r_f, NIS, Utility, ALL_utility)
                    # print(r_f, vss)
                if (Uc + vss) <= Min_utility:
                    no_hope.append(Can_pattern) #  添加进no_hope后就从候选模式中删除了 不在参加后续的链接操作
        C_k_1 = C_k
        k += 1


    end_time = time.time()

    H = []
    for k in High_utility.keys():
        if len(k) > 1:
            H.append(k)

    print(get_memory_usage())

    print(H, "高效用同位模式")  # 高效用模式输出
    print(len(H), "高效用模式个数")

    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")


if __name__ == "__main__":
    my_func()
