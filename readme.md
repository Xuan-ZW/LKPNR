
1. data sampling
cd NNR
bash download_extract_MIND.sh.


2. Data process

2.1. LLM_embedding_dic  #The Embedding by LLM
./NNR/pretrain_emb/item_emb.npy  

2.2.  linked_entity_dic #Save the adjacent entity dictionary, leaving only adjacent entities with emb vectors
./graph/count_link_count.ipynb  

1.3 other #Saving entities appearing in news&embedding entities in merged datasets

./graph/get_node_emb.py    
./graph/get_node_emb.ipynb 



2 model training
python  main.py --news_encoder=MHSA --user_encoder=MHSA 



