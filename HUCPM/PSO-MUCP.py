import gc
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
Utility = {'A': 2, 'B': 4, 'C': 8, 'D': 4, "E": 1, "F": 4, "G": 5, "H": 3, "I": 9, "J": 3, 'K': 5, 'L': 10,'M': 8}
Min_utility = 0.5
D = 1300 # 距离阈值越大生成得邻近关系越多

start_time = time.time()


#  计算出每个特征的总效用然后将特征按照总效用从小到大排列
def Compute_utility(Utility, instance):
    utility_dict = {}
    for elem in Utility.keys():
        number_instance = 0
        for i in instance:
            if i[0][0] == elem:
                number_instance += 1
        utility_dict[elem] = number_instance * Utility[elem]
    sorted_utility = dict(sorted(utility_dict.items(), key=lambda x: x[1]))
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


#  计算粗略上界并剪枝
def Prune_one(nis, utility, min_utility, ALL):
    positive = []  # 存放正效用特征
    negative = []  # 存放负效用特征
    Ins_number = {}
    Up_number = {}
    for nis_ket in nis.keys():
        if nis_ket[0] not in Ins_number.keys():
            Ins_number[nis_ket[0]] = []
            Up_number[nis_ket[0]] = 0
            if utility[nis_ket[0]] > 0:
                positive.append(nis_ket[0])
            else:
                negative.append(nis_ket[0])
        Ins_number[nis_ket[0]] = list(set(Ins_number[nis_ket[0]]) | set(nis[nis_ket]))
        Ins_number[nis_ket[0]].append(nis_ket)

    for item in Ins_number.keys():
        for item_elem in Ins_number[item]:
            if utility[item_elem[0]] > 0:
                Up_number[item] += utility[item_elem[0]]
    #  print(Up_number)
    #  print(Ins_number)
    #  删除上界小于min的特征
    UP_del = Up_number.copy()
    delete = []
    for up_key in Up_number:
        if Up_number[up_key] / ALL <= min_utility:
            del UP_del[up_key]
            delete.append(up_key)
    #  print(UP_del, delete)
    positive_hash = {}
    negative_hash = {}
    for del_key in UP_del.keys():
        if del_key in positive:
            positive_hash[del_key] = UP_del[del_key]
        else:
            negative_hash[del_key] = UP_del[del_key]
    p_sorted_keys = sorted(positive_hash, key=positive_hash.get)
    n_sorted_keys = sorted(negative_hash, key=negative_hash.get)
    sort_list = p_sorted_keys + n_sorted_keys

    delete_nis = {}
    for key in nis.keys():
        if key[0] not in delete:
            delete_nis[key] = []
            for instance in nis[key]:
                if instance[0] not in delete:
                    delete_nis[key].append(instance)
    #  print(p_sorted_keys)
    #  print(n_sorted_keys)
    #  print(sort_list)
    return sort_list, delete, positive, negative, delete_nis


# print(Sort_list, Delete)


def Enum_Cliques(nis, delete):
    clique = {}
    for key in nis:
        if key[0] in delete:
            continue
        clique[key] = []
        flag_list = nis[key].copy()  # 可以优化为直接遍历，避免复制

        while flag_list:
            current = flag_list.pop(0)  # 移除当前节点
            Can_key = [current]

            # 优化交集计算：避免重复转换 set 和 list
            neighbors = set(nis[current])
            inter = [x for x in flag_list if x in neighbors]

            # 改用迭代DFS代替递归
            stack = [(Can_key, inter)]
            while stack:
                can_key, inter = stack.pop()
                if not inter:
                    clique[key].append(can_key)
                    continue
                if len(inter) == 1:
                    clique[key].append(can_key + [inter[0]])
                    continue

                # 避免频繁的列表拷贝
                first_elem = inter[0]
                new_can_key = can_key + [first_elem]
                remaining = inter[1:]

                # 计算新的交集
                new_neighbors = set(nis[first_elem])
                new_inter = [x for x in remaining if x in new_neighbors]

                stack.append((new_can_key, new_inter))
                stack.append((can_key, remaining))

    return clique


#  生成哈希表
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


#  输入一个列表，生成该列表里的模式组合而成的k+1阶模式
def gener_k(compute_list, sort_list, min_utility, no_hope, negative, hash_list, all_utility):
    can_hash = {}
    for k_1_can_key in compute_list:
        if compute_list[k_1_can_key][0] / all_utility < min_utility:  # 它的超集不可能是高效用同位模式
            no_hope.append(k_1_can_key)  # 存储没有希望的模式，在后面的侯选中就不生成这种没有希望的模式
            continue
        for i in range(len(sort_list)):
            if compute_list[k_1_can_key][1] / all_utility <= min_utility and sort_list[
                i] in negative:  # 如果k-1阶模式是非高效用模式，则它加上负效用的超集也可以直接删减
                continue

            #  现在要确保生成的键是有序的
            x = 0
            for j in range(len(sort_list)):
                if k_1_can_key[len(k_1_can_key) - 1] == sort_list[j]:
                    x = j
                    break
            if x >= i:
                continue
            if sort_list[i] in k_1_can_key:
                continue
            k_can_key = k_1_can_key + sort_list[i]

            #  现在要检查生成的k阶模式是否包含已经被剪枝的元素
            flag = 0
            for elem_1 in no_hope:
                if k_can_key[:len(elem_1)] == elem_1:
                    flag = 1
                    break
            if flag == 1:
                continue

            #  现在要检查生成的k阶模式是否是存在的组合：
            flag_1 = 0
            for elem_2 in hash_list:
                letters_sets = set(k_can_key)
                if all(letter in elem_2 for letter in letters_sets):
                    flag_1 = 1  # 如果包含在某一个键里说明存在
                    break
            if flag_1 == 0:
                continue

            can_hash[k_can_key] = []
    return can_hash


#  查找顺序
def order(sort_list, elem_1, elem_2):
    s1 = -1
    s2 = -1
    for i in range(len(sort_list)):
        if sort_list[i] == elem_2[len(elem_2) - 1]:
            s1 = i
        if sort_list[i] == elem_1:
            s2 = i
    if s2 > s1:
        return True
    return False


# print(ALL_utility)


#  负效用定理：1.全是负效用的特征可以直接删除 2.如果k-1阶模式是非高效用模式，则它加上负效用的超集也可以直接删减
#  计算列表
#  精确上界剪枝
#  找到高效用同位模式
def Find_High_utility_pattern(hash_table, sort_list, min_utility, negative, utility, ALL):
    all_pattern_num = 0

    high_utility = []
    hash_key_list = list(hash_table.keys())
    No_hope = []
    Can_hash = {}

    for can_key in sort_list:
        if can_key not in negative:  # 如果特征全是负效用就跳过，因为我们是有序的模式，如果一个模式全是负效用，那么他后面也不会再加上正效用了
            Can_hash[can_key] = [0, 0]

    while len(Can_hash.keys()) != 0:
        number_hash_1 = {}
        up_number_1 = {}
        for elem in Can_hash:
            number_hash_1[elem] = {}
            up_number_1[elem] = {}
        #  计算列表
        for hash_key in hash_table.keys():
            for Can_hash_key in Can_hash:
                number_hash = {}  # 找到不重复的实例个数
                up_number = {}  # 找超集不重复的实例个数
                letters_set = set(Can_hash_key)
                if all(letter in hash_key for letter in letters_set):  # 证明这个hash键里包含这个候选的所有字符
                    for elem_key in hash_table[hash_key]:
                        if elem_key in Can_hash_key:
                            number_hash[elem_key] = []
                            if elem_key not in number_hash_1[Can_hash_key]:
                                number_hash_1[Can_hash_key][elem_key] = []
                            for elem_3 in hash_table[hash_key][elem_key]:
                                if elem_3 not in number_hash_1[Can_hash_key][elem_key]:
                                    number_hash[elem_key].append(elem_3)

                        if elem_key not in up_number and (elem_key in Can_hash_key or order(sort_list, elem_key,
                                                                                            Can_hash_key)) and elem_key not in negative:
                            up_number[elem_key] = []
                            if elem_key not in up_number_1[Can_hash_key]:
                                up_number_1[Can_hash_key][elem_key] = []
                        if (elem_key not in Can_hash_key and not order(sort_list, elem_key,
                                                                       Can_hash_key)) or elem_key in negative:
                            continue
                        for elem in hash_table[hash_key][elem_key]:
                            if hash_key != Can_hash_key and elem not in up_number_1[Can_hash_key][elem_key]:
                                up_number[elem_key].append(elem)

                if len(number_hash_1[Can_hash_key]) == 0 and len(up_number_1[Can_hash_key]) == 0:
                    number_hash_1[Can_hash_key] = number_hash
                    up_number_1[Can_hash_key] = up_number
                else:
                    for number_hash_elem in number_hash:
                        for h in number_hash[number_hash_elem]:
                            number_hash_1[Can_hash_key][number_hash_elem].append(h)
                    for up_number_elem in up_number:
                        for j in up_number[up_number_elem]:
                            up_number_1[Can_hash_key][up_number_elem].append(j)
                #  计算上界值和真实值
                # print(number_hash,"nh")
                # print(Can_hash, "cccccccc")
                for s in number_hash:
                    Can_hash[Can_hash_key][1] += len(number_hash[s]) * utility[s]
                for t in up_number:
                    Can_hash[Can_hash_key][0] += len(up_number[t]) * utility[t]
        # for elem in Can_hash.keys():
        # print(Can_hash)
        # print(No_hope)

        for key in Can_hash:
            # print(key, (Can_hash[key][1] / ALL), Can_hash[key][1], ALL)
            if (Can_hash[key][1] / ALL) >= min_utility:
                # print(key, (Can_hash[key][1] / ALL))
                high_utility.append(key)
        # print(Can_hash, "CAN_HASH")  # 打开这个print可以看到具体的上界数值和具体的效用======================================
        Can_hash = gener_k(Can_hash, sort_list, min_utility, No_hope, negative, hash_key_list, ALL)
        #  初始化
        for can_key in Can_hash:
            Can_hash[can_key] = [0, 0]

        all_pattern_num = all_pattern_num + len(Can_hash.keys())

    return high_utility, No_hope, all_pattern_num


# 替代方案示例 - 手动检查内存
import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def my_func():

    S_1, ALL_utility = Compute_utility(Utility, Instance)
    NIS = grid_method(Instance, D)
    Sort_list, Delete, Positive, Negative, deleted_ins = Prune_one(NIS, Utility, Min_utility, ALL_utility)
    print(Sort_list, Delete)
    # print(deleted_ins)

    Clique = Enum_Cliques(deleted_ins, Delete)
    # print(Clique)
    print(f"Clique size: {sum(len(v) for v in Clique.values())}")  # 打印Clique大小

    Hash_table = con_hash_table(Clique)
    print(Hash_table)
    del Clique  # 显式删除Clique
    gc.collect()  # 强制垃圾回收

    h, N, all_pattern = Find_High_utility_pattern(Hash_table, Sort_list, Min_utility, Negative, Utility, ALL_utility)
    end_time = time.time()

    H = [k for k in h if len(k) > 1]

    print(get_memory_usage())

    print(H, "高效用模式")
    print(len(H), "高效用模式的个数")
    print(f"程序运行时间: {end_time - start_time} 秒")

if __name__ == "__main__":
    my_func()
