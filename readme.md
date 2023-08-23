
# 1. data sampling
cd NNR

bash download_extract_MIND.sh.


# 2. Data process

## 2.1. The Embedding by LLM
./NNR/pretrain_emb/item_emb.npy  

## 2.2. Save the adjacent entity dictionary, leaving only adjacent entities with emb vectors
./graph/count_link_count.ipynb  

## 2.3 Saving entities appearing in news&embedding entities in merged datasets

./graph/get_node_emb.py    
./graph/get_node_emb.ipynb 



# 3. model training
python  main.py --news_encoder=MHSA --user_encoder=MHSA 

# 4. Ohters
The necessary files for running the program can be downloaded from https://pan.baidu.com/s/13AVXSWxIuXI14UgKjaSpbg?pwd=siuf




