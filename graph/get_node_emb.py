import pandas as pd
import pickle
def get_entity_in_news(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split('\t')
        title_entity = parts[-2]
        abstract_entity = parts[-1]
        data.append([title_entity, abstract_entity])
    node_in_train = pd.DataFrame(data, columns=["title_entity", "abstract_entity"])

    # 连接title_entity和abstract_entity
    node_in_train_all = pd.concat([node_in_train['title_entity'], node_in_train['abstract_entity']], ignore_index=True)
    node_in_train_all = pd.DataFrame(node_in_train_all, columns=['node_in_train_all'])

    # 对每一行进行拆分
    new_rows = []
    for _, row in node_in_train_all.iterrows():
        for item in eval(row['node_in_train_all']):
            new_rows.append(item)
    new_df = pd.DataFrame(new_rows)

    # 拆分后去重
    new_df.drop_duplicates(subset="WikidataId", keep="first", inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    return new_df

def get_entity_in_emb_file():
    # 读取entity_embedding.vec文件
    with open("../MIND-200k/train/entity_embedding.vec", "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 解析每一行，提取KEY和Value
    data = []
    for line in lines:
        parts = line.strip().split()
        key = parts[0]
        value = [float(v) for v in parts[1:]]
        data.append([key, value])

    node_emb_train = pd.DataFrame(data, columns=["Node", "Embedding"])

    # 读取entity_embedding.vec文件
    with open("../MIND-200k/dev/entity_embedding.vec", "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 解析每一行，提取KEY和Value
    data = []
    for line in lines:
        parts = line.strip().split()
        key = parts[0]
        value = [float(v) for v in parts[1:]]
        data.append([key, value])

    node_emb_dev = pd.DataFrame(data, columns=["Node", "Embedding"])

    node_emb = pd.concat([node_emb_train, node_emb_dev])
    node_emb.drop_duplicates(subset="Node", keep="first", inplace=True)
    # 重置索引
    node_emb.reset_index(drop=True, inplace=True)
    return node_emb

if __name__ == "__main__":
    # 获得数据集中出现过的entity
    node_in_train = get_entity_in_news('../MIND-200k/train/news.tsv')
    node_in_dev = get_entity_in_news('../MIND-200k/dev/news.tsv')
    node_in_test = get_entity_in_news('../MIND-200k/test/news.tsv')

    node_all = pd.concat([node_in_train, node_in_dev], ignore_index=True)
    node_all = pd.concat([node_all, node_in_test], ignore_index=True)
    node_all.drop_duplicates(subset="WikidataId", keep="first", inplace=True)
    node_all.reset_index(drop=True, inplace=True)
    print(node_all)

    # 获取entity库
    entity_in_emb_file = get_entity_in_emb_file()
    print(entity_in_emb_file)


    # 将df1和df2中的节点转换为集合
    set_df1_nodes = set(node_all['WikidataId'])
    set_df2_nodes = set(entity_in_emb_file['Node'])
    # 检查未包含节点
    nodes_not_in_df2 = set_df1_nodes - set_df2_nodes
    print(len(nodes_not_in_df2))

    # 保存set到本地文件
    #with open('entity_in_news.pickle', 'wb') as file:
        #pickle.dump(set_df1_nodes, file)
    
    #with open('entity_in_emb_file.pickle', 'wb') as file:
        #pickle.dump(set_df2_nodes, file)
        