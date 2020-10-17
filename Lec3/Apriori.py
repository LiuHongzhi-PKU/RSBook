<<<<<<< HEAD
import pandas as pd
def load_data_set():
    #载入数据
    # 数据集
    data_path = "../u.data"
    df = pd.read_csv(data_path, header=None, sep="	")
    df.columns = ["user", "movie", "rate", "time"]
    dataset = dict()
    # 整理数据，将mvlen数据整理成字典，字典的键是用户名，值的该用户评价超过3的电影集合
    for i in range(0, df.shape[0]):
        if df.loc[i, "user"] not in dataset:
            dataset[df.loc[i, "user"]] = set()
        if df.loc[i, "rate"] >= 3:
            dataset[df.loc[i, "user"]].add(df.loc[i, "movie"])
    data_set=[]
    for i,j in dataset.items():
        data_set.append(j)
    print(data_set)
    return data_set


def create_C1(data_set):
    #频繁一项集创建
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1


def is_apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    #频繁k项集创建
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk


def generate_L(data_set, k, min_support):
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data


def generate_big_rules(L, support_data, min_conf):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list

#主函数
if __name__ == "__main__":
    #把movielens数据集转换成以用户为单位进行组织的数据集
    data_set = load_data_set()
    #得到各种物品组合的支持度
    L, support_data = generate_L(data_set, k=3, min_support=0.2)
    #得到关联规则
    big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)
    #输出查看
    for Lk in L:
        for freq_set in Lk:
            print (freq_set, support_data[freq_set])
    print ("关联规则")
    for item in big_rules_list:
        print (item[0], "=>", item[1], "conf(置信度): ", item[2])
=======
import pandas as pd
def load_data_set():
    #载入数据
    # 数据集
    data_path = "../u.data"
    df = pd.read_csv(data_path, header=None, sep="	")
    df.columns = ["user", "movie", "rate", "time"]
    dataset = dict()
    # 整理数据，将mvlen数据整理成字典，字典的键是用户名，值的该用户评价超过3的电影集合
    for i in range(0, df.shape[0]):
        if df.loc[i, "user"] not in dataset:
            dataset[df.loc[i, "user"]] = set()
        if df.loc[i, "rate"] >= 3:
            dataset[df.loc[i, "user"]].add(df.loc[i, "movie"])
    data_set=[]
    for i,j in dataset.items():
        data_set.append(j)
    print(data_set)
    return data_set


def create_C1(data_set):
    #频繁一项集创建
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1


def is_apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    #频繁k项集创建
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk


def generate_L(data_set, k, min_support):
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data


def generate_big_rules(L, support_data, min_conf):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list

#主函数
if __name__ == "__main__":
    #把movielens数据集转换成以用户为单位进行组织的数据集
    data_set = load_data_set()
    #得到各种物品组合的支持度
    L, support_data = generate_L(data_set, k=3, min_support=0.2)
    #得到关联规则
    big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)
    #输出查看
    for Lk in L:
        for freq_set in Lk:
            print (freq_set, support_data[freq_set])
    print ("关联规则")
    for item in big_rules_list:
        print (item[0], "=>", item[1], "conf(置信度): ", item[2])
>>>>>>> 0718e6ad2a69a791b75deeceab0930001feddcae
