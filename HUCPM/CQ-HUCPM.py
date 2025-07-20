import gc
import math
import os
import tracemalloc
from math import sqrt
import time
import psutil


f = open(r"D:\PSO-MUCP 代码\IDEA co-location\数据文档\California_POI 13f.csv", "r",
         encoding="UTF-8")  # AA.text BB.text CC.text
Instance_utils = {'A': 2, 'B': 4, 'C': 8, 'D': 4, "E": 1, "F": 4, "G": 5, "H": 3, "I": 9, "J": 3, 'K': 5, 'L': 10,'M': 8}
Instance_util = {}
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
        Instance_util[s] = Instance_utils[temp[0]]
f.close()
D = 1300
Min_util = 0.5

start_time = time.time()
"=================================================================================================================="


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


#  对数据进行归一化处理
# def Normalized_util(instance_util):
#     #  先找到每个特征对应实例效用的最大值
#     max_util = {}
#     for key_word in instance_util:
#         if key_word[0] not in max_util.keys():
#             max_util[key_word[0]] = instance_util[key_word]
#         if max_util[key_word[0]] < instance_util[key_word]:
#             max_util[key_word[0]] = instance_util[key_word]
#
#     instance_util_1 = instance_util.copy()
#     print(max_util)
#     for key in instance_util_1.keys():
#         instance_util[key] = math.log(instance_util_1[key], 10) / math.log(max_util[key[0]], 10)
#     return instance_util


#  计算每个特征的总效用
def compute_util(normal_util):
    util = {}
    all_utility = 0
    for key in normal_util:
        all_utility += normal_util[key]
        if key[0] not in util.keys():
            util[key[0]] = round(normal_util[key], 3)
        else:
            util[key[0]] += round(normal_util[key], 3)
    return util, all_utility


#  枚举空间实例团
def digui(can_key, inter, nis, A):
    # print(can_key, "can_key")
    # print(inter, "inter")
    if len(inter) == 0:
        A.append(can_key)
        return
    if len(inter) == 1:
        A.append(can_key + [inter[0]])
        return
    pre_inter = inter.copy()
    for elem in inter:
        pre_inter.pop(0)
        new_can_key = can_key + [elem]
        new_inter = sorted(list(set(nis[elem]) & set(pre_inter)))
        digui(new_can_key, new_inter, nis, A)


def is_element_in_nested_list(lst, element):
    for sublist in lst:
        if sublist is not None:
            if element in sublist:
                return True
    return False


def Enum_Cliques(nis):
    clique = {}
    for key in nis:
        clique[key] = []
        flag_list = nis[key].copy()

        while len(flag_list) != 0:
            Can_key = [flag_list[0]]
            inter = sorted(list(set(nis[flag_list[0]]) & set(flag_list)))
            flag_list.remove(flag_list[0])

            A = []
            digui(Can_key, inter, nis, A)

            if len(A) != 0:
                for elem in A:
                    clique[key].append(elem)
            else:
                s = Can_key
                if not is_element_in_nested_list(clique[key], Can_key[0]):
                    clique[key].append(s)
                if Can_key != s:
                    clique[key].append(s)
    return clique


# print(Clique)


#  构建哈希表
def con_hash_table(clique):
    hash_table = {}  # 哈希表
    for key in clique:
        for clique_list in clique[key]:
            if clique_list is None:
                continue

            # 构建哈希键
            hash_key = key[0] + ''.join(elem[0] for elem in clique_list)
            hash_key = ''.join(sorted(hash_key))

            # 创建哈希表中的内层哈希表
            if hash_key not in hash_table:
                hash_table[hash_key] = {}
                hash_table[hash_key][key[0]] = [key]
                for elem in clique_list:
                    hash_table[hash_key][elem[0]] = [elem]
            else:
                # 添加关键字实例和特征实例
                if key[0] not in hash_table[hash_key]:
                    hash_table[hash_key][key[0]] = [key]
                else:
                    if key not in hash_table[hash_key][key[0]]:
                        hash_table[hash_key][key[0]].append(key)
                for elem in clique_list:
                    if elem[0] not in hash_table[hash_key]:
                        hash_table[hash_key][elem[0]] = [elem]
                    else:
                        if elem not in hash_table[hash_key][elem[0]]:
                            hash_table[hash_key][elem[0]].append(elem)
    return hash_table


#  计算自适应UPI
def compute_adaptive_UPI(key, hash_table, normal_util, util, all_s):
    # print(hash_table.keys())
    # print(key)

    un_repeat = {}
    UPI = []

    for z in range(len(key)):
        un_repeat[key[z]] = []

    for table_key in hash_table:
        s = 0
        for i in range(len(key)):
            if key[i] not in table_key:
                s = 1
        if s == 0:
            for j in range(len(key)):
                for elem in hash_table[table_key][key[j]]:
                    un_repeat[key[j]].append(elem)

    Un_repeat = {}
    for keys in un_repeat:
        Un_repeat[keys] = list(set(un_repeat[keys]))
    # print(normal_util)
    # print(all_s)
    #  计算参与实例的效用
    z_utility = 0

    for key in Un_repeat:
        for elem in Un_repeat[key]:
            z_utility += normal_util[elem]
    ture_utility = z_utility / all_s
    return ture_utility
    # for i in range(len(key)):
    #     q = 0
    #     for elem in Un_repeat[key[i]]:
    #         q += normal_util[elem]  # q就是特征的参与实例的效用
    #     UR_intra = (q / util[key[i]])  # 除以该特征的总效用就可以UR——intra
    #
    #     if len(key) != 1:
    #         y = 0  # 剩余特征的参与效用
    #         z_y = 0  # 剩余特征的总效用
    #         for j in range(len(key)):
    #             if key[j] != key[i]:
    #                 for elem in Un_repeat[key[j]]:
    #                     y += normal_util[elem]
    #                 z_y += util[key[j]]
    #         UR_inter = (y / z_y)
    #         aerfa = round(UR_intra / (UR_inter + UR_intra), 3)
    #         beita = round(UR_inter / (UR_inter + UR_intra), 3)
    #         UPR_C = round(aerfa * UR_intra + beita * UR_inter, 3)
    #         UPI.append(UPR_C)
    #     else:
    #         return UR_intra
    # return min(UPI)


# 生成所有子集
def generate_set(pattern):
    all_sub = []
    sub_1_set = []
    for i in range(len(pattern)):
        sub_1_set.append(pattern[i])
        all_sub.append(pattern[i])  # 将一阶同模式加入候选模式中

    k = 1
    sub_set_2 = sub_1_set.copy()
    while k < len(pattern) - 1:
        sub_k_1 = []
        for i in range(len(sub_set_2)):
            temp = []
            for j in range(len(sub_1_set)):
                if sub_1_set[j] not in sub_set_2[i]:
                    temp.append(sub_set_2[i] + sub_1_set[j])

            for x in temp:
                sub_k_1.append(''.join(sorted(x)))
                all_sub.append(''.join(sorted(x)))
            sub_k_1 = list(set(sub_k_1))
        sub_set_2 = sub_k_1
        k += 1
    all_sub = list(set(all_sub))
    return all_sub


#  输出所有的高效用模式算法
def CQ_Hucpm(hash_table, min_util, normal_util, util, all_u):
    HUCPS = {}
    candid_pattern = list(hash_table.keys())

    for can in candid_pattern:
        PI = compute_adaptive_UPI(can, hash_table, normal_util, util, all_u)
        if PI >= min_util:
            s = "".join(sorted(can))
            HUCPS[s] = PI

        Sub_set = generate_set(can)
        for pattern in Sub_set:
            if pattern not in HUCPS:
                Sub_pi = compute_adaptive_UPI(pattern, hash_table, normal_util, util, all_u)
                if Sub_pi >= min_util:
                    h = "".join(sorted(pattern))
                    HUCPS[h] = Sub_pi

    return HUCPS

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# @profile
def my_func():
    # 开始内存跟踪
    tracemalloc.start()

    NIS = grid_method(Instance, D)
    # print(NIS)
    # Normal_instance_util = Normalized_util(Instance_util)
    Util, ALL = compute_util(Instance_util)
    Clique = Enum_Cliques(NIS)
    Hash_table = con_hash_table(Clique)

    del Clique
    gc.collect()  # 强制垃圾回收

    # print(Hash_table)
    HT = CQ_Hucpm(Hash_table, Min_util, Instance_util, Util, ALL)
    end_time = time.time()

    H = []
    for k in HT.keys():
        if len(k) > 1:
            H.append(k)

    print(get_memory_usage())

    print(H, "高效用模式")
    print(len(H), "高效用模式个数")

    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")


if __name__ == "__main__":
    my_func()
