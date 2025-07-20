import math
import time
from math import sqrt

f = open(r"D:\PSO-MUCP 代码\IDEA co-location\数据文档\California_POI 13f.csv", "r",
         encoding="UTF-8")  # AA.text BB.text CC.text
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
utility = {'A': 2, 'B': 4, 'C': 8, 'D': 4, "E": 1, "F": 4, "G": 5, "H": 3, "I": 9, "J": 3, 'K': 5, 'L': 10,'M': 8}
min_utility = 0.5
d = 1300  # 距离阈值越大生成得邻近关系越多

start_time = time.time()


#  计算出每个特征的总效用然后将特征按照总效用从大到小排列
def Compute_utility(Utility, instance):
    utility_dict = {}
    for elem in Utility.keys():
        number_instance = 0
        for i in instance:
            if i[0][0] == elem:
                number_instance += 1
        utility_dict[elem] = number_instance * Utility[elem]
    sorted_utility = dict(sorted(utility_dict.items(), key=lambda x: x[1], reverse=True))
    all_utility = 0
    for elem in sorted_utility.keys():
        all_utility += sorted_utility[elem]
    return sorted_utility, all_utility


#  计算出所有的实例邻近关系
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


#  构建搜索树
def search_tree(ns, s_utility):
    tree = {}
    for f_elem in s_utility.keys():
        temp = {}
        for nsf_elem in ns.keys():
            if nsf_elem[0] == f_elem and len(ns[nsf_elem]) != 0:
                dict_ns = {}
                for elem in ns[nsf_elem]:
                    if elem[0] in dict_ns.keys():
                        dict_ns[elem[0]].append(elem)
                    else:
                        dict_ns[elem[0]] = []
                        dict_ns[elem[0]].append(elem)
                temp[nsf_elem] = dict_ns
        tree[f_elem] = temp
    return tree


#  生成一个模式的行实例
def CS_HBS(c, tree, ns):
    fake_row_instance = []
    for instance_dict in tree[c[0]]:
        temp = [[instance_dict]]
        my_string = ''.join([item for item in list(tree[c[0]][instance_dict].keys())])
        flag = 0
        for i in range(1, len(c)):
            if c[i] not in my_string:
                flag = 1
        if flag == 0:
            for i in range(1, len(c)):
                temp_s = []
                for elem in tree[c[0]][instance_dict][c[i]]:  #
                    for elem_list in temp:  # 将每一个存在的行实例都加上这个实例
                        temp_2s = elem_list.copy()
                        temp_2s.append(elem)
                        temp_s.append(temp_2s)
                temp = temp_s
        else:
            temp = []
        if len(temp):
            for elem in temp:
                fake_row_instance.append(elem)
    #  验证是否是真的行实例
    true_row_instance = []
    # print(c, fake_row_instance)
    for elem in fake_row_instance:
        flag = 0
        for i in range(1, len(elem) - 1):
            if not all(element in ns[elem[i]] for element in elem[i + 1: len(elem)]):
                flag = 1
        if flag == 0:
            true_row_instance.append(elem)
    # print(true_row_instance)
    return true_row_instance


#  计算模式的效用
def compute_utility(c, Utility, ture_row_instance):
    allocation = {}
    for row_elem in ture_row_instance:
        for elem in row_elem:
            if elem[0] in allocation.keys():
                if elem not in set(allocation[elem[0]]):
                    allocation[elem[0]].append(elem)
            else:
                allocation[elem[0]] = []
                allocation[elem[0]].append(elem)
    pattern_utility = 0
    if len(allocation.keys()):
        for str_elem in c:
            s = Utility[str_elem] * len(allocation[str_elem])
            pattern_utility += s
    return pattern_utility


#  找到扩展模式集
def find_pattern_set(c, tree):
    pattern_set = {}
    for element in tree[c[0]]:
        for in_key in tree[c[0]][element]:
            if in_key not in pattern_set.keys():
                pattern_set[in_key] = []
                for item in tree[c[0]][element][in_key]:
                    if item not in pattern_set[in_key]:
                        pattern_set[in_key].append(item)
            else:
                for item in tree[c[0]][element][in_key]:
                    if item not in pattern_set[in_key]:
                        pattern_set[in_key].append(item)
    return pattern_set


#  损失效用的计算
def luc(c, s_utility, pattern_utility, all_utility):
    temp_utility = 0
    for str_elem in c:
        temp_utility += s_utility[str_elem]
    luc_lv = (temp_utility - pattern_utility) / all_utility
    return luc_lv


# 计算模式效用上界
def ubc(pattern_utility, pattern_set, all_utility, Utility):
    temp_ubc = 0
    for key_elem in pattern_set.keys():
        temp = Utility[key_elem] * len(pattern_set[key_elem])
        temp_ubc += temp
    # print("========================================")
    # print(pattern_set)
    # print(temp_ubc)
    # print(pattern_utility)
    euc = (temp_ubc + pattern_utility) / all_utility
    # print(euc)
    # print("+++++++++++++++++++++++++++++++++++++++=")
    return euc


#  找到所有的候选模式
def search(s_utility):
    list_al = []  # 存储一阶模式
    for keyword in s_utility.keys():
        list_al.append(keyword)

    # 生成二阶候选模式
    pattern_list = list_al.copy()
    for i in range(len(list_al)):
        can_s = ""
        for j in range(i + 1, len(list_al)):
            can_s = list_al[i] + list_al[j]
            pattern_list.append("".join(sorted(can_s)))

    # 生成k阶候选模式
    temp = pattern_list.copy()
    while True:
        temp_now = []
        for elem in temp:
            s_t = ""
            for al in list_al:
                if al not in elem:
                    s_t = "".join(sorted(elem + al))
                    temp_now.append(s_t)
        for elem in temp_now:
            if elem not in pattern_list:
                pattern_list.append(elem)
        if len(temp_now) == 0:
            break
        else:
            temp = temp_now
    return pattern_list

# 替代方案示例 - 手动检查内存
import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def my_func():

    Ns = grid_method(Instance, d)
    S_utility, All_utility = Compute_utility(utility, Instance)
    TREE = search_tree(Ns, S_utility)

    pattern_num = 0
    pruning = 0
    high_pattern = []
    non_pattern = []
    list_al = []  # 存储一阶模式
    for keyword in S_utility.keys():
        list_al.append(keyword)
        if S_utility[keyword] / All_utility >= min_utility:
            high_pattern.append(keyword)

    # 生成二阶候选模式
    pattern_list = []
    for i in range(len(list_al)):
        can_s = ""
        for j in range(i + 1, len(list_al)):
            can_s = list_al[i] + list_al[j]
            pattern_list.append("".join(sorted(can_s)))

    pattern_num += len(pattern_list)
    # Even_pattern = search(S_utility)
    while len(pattern_list) != 0:
        Even_pattern = pattern_list
        print(Even_pattern)
        for can_pattern in Even_pattern:
            C = can_pattern
            if len(non_pattern) != 0:  # 只要有一个无希望模式在候选模式中就跳过这个候选模式
                Flag = 0
                for elem in non_pattern:
                    flag = 0
                    for al in elem:
                        if al not in C:
                            flag = 1
                            continue
                    if flag == 0:  # 说明是候选模式的超集
                        Flag = 1
                        continue
                if Flag == 1:
                    pruning += 1
                    continue

            Ture_row_instance = CS_HBS(C, TREE, Ns)
            Pattern_utility = compute_utility(C, utility, Ture_row_instance) / All_utility
            if Pattern_utility >= min_utility:
                # print(C, Pattern_utility)
                high_pattern.append(C)
            else:
                Luc_1 = luc(C, S_utility, compute_utility(C, utility, Ture_row_instance), All_utility)
                if Luc_1 <= 1 - min_utility:
                    Pattern_set = find_pattern_set(C, TREE)
                    euc_1 = ubc(compute_utility(C, utility, Ture_row_instance), Pattern_set, All_utility, utility)
                    if euc_1 <= min_utility:
                        non_pattern.append(C)
        # 生成k阶候选模式
        temp = pattern_list.copy()
        pattern_list = []
        for elem in temp:
            s_t = ""
            for al in list_al:
                if al not in elem:
                    s_t = "".join(sorted(elem + al))
                    if s_t not in pattern_list:
                        pattern_list.append(s_t)
        pattern_num += len(pattern_list)

    end_time = time.time()

    H = []
    for k in high_pattern:
        if len(k) > 1:
            H.append(k)

    print(get_memory_usage())

    print(H, "高效用同位模式")  # 高效用模式输出
    print(len(H), "高效用模式个数")

    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time} 秒")


if __name__ == "__main__":
    my_func()
