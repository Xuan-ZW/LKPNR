import pandas as pd
import numpy as np
from transformers import AutoTokenizer,AutoModel
import torch
from tqdm import tqdm
news = pd.read_csv(
    "./merged_large_small_item.tsv", 
    sep="\t",
    names=["itemID","category","subcategory","title","abstract","url","title_entities","abstract_entities"])
def construct_news_text(x):
    return "news title : " + str(x["title"]) + "\n" + "news category : " + str(x["category"]) + "\n" + "news abstract : " + str(x["abstract"])
news["news_text"] = news.apply(lambda x : construct_news_text(x),axis = 1)
itemid_list = news["itemID"].tolist()
itemtext_list = news["news_text"].tolist()

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("../glm2", trust_remote_code=True)
model = AutoModel.from_pretrained("../glm2", trust_remote_code=True).float().to(device)

# freeze
for param in model.parameters():
    param.requires_grad = False
model.config.output_hidden_states=True


item_text_map = {}
for idx,item_text in tqdm(enumerate(itemtext_list)):
    itemid = itemid_list[idx]
    inputs = tokenizer.encode_plus(item_text, return_tensors="pt", padding=True).to(device=model.device)
    hidden_states = model.forward(input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'], 
                                return_dict=True).hidden_states[-1].detach().cpu().permute(1,0,2) # 取最后一个隐藏层
    # 按照第二个维度求和
    sum_array = np.mean(hidden_states.numpy(), axis=1)
    # 将维度变为(4096)
    reshaped_array = sum_array.reshape(4096)
    item_text_map[itemid] = reshaped_array


np.save("item_emb.npy",item_text_map)