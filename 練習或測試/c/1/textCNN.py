from datasets import load_dataset
import os
import time
from collections import Counter
from multiprocessing import Process,Queue,current_process

def pre_process(data):
    words_list=[]
    counts_list=[]
    for sentence in data["text"]:
        c = Counter(sentence.split(" "))
        words_list.append(list(c.keys()))
        counts_list.append(list(c.values()))
    return {"words": words_list, "counts": counts_list}

def queue_put(words_col,counts_col,idx_queue,queue):
    wall_start=time.perf_counter()
    put_time=0.0
    n_chunks=0
    idx=-1
    while True:
        idx=idx_queue.get()
        if idx*1000>=len(words_col):
            idx_queue.put(idx)
            break
        idx_queue.put(idx+1)
        chunk_voc=set()
        chunk_counter=Counter()
        for i in range(idx*1000,min(len(words_col),(idx+1)*1000)):
            data=dict(zip(words_col[i],counts_col[i]))
            chunk_voc.update("".join(data.keys()))
            chunk_counter.update(data)
        t0=time.perf_counter()
        queue.put((chunk_voc,chunk_counter))
        put_time+=time.perf_counter()-t0
        n_chunks+=1
    queue.put(None)
    wall=time.perf_counter()-wall_start
    print(f"[producer {current_process().pid}] chunks={n_chunks} wall={wall:.3f}s "
          f"put_time={put_time:.3f}s ({100*put_time/wall:.1f}% blocked on IPC)")

def count_substr(counter,queue):
    pass
def change_key(counter,target):
    pass

# map的回傳列數需等於該次處理資料數
if __name__ == "__main__":
    # 載入 IMDB 資料集
    dataset = load_dataset("stanfordnlp/imdb")

    # 查看資料結構
    print(dataset)
    # 可以透過這樣取得訓練集與測試集
    train_data = dataset['train']
    test_data = dataset['test']

    # 印出一筆範例看看
    print("文字:", train_data[0]['text'])
    print("標籤 (0代表負評, 1代表正評):", train_data[0]['label'])

    CPU=max(1, os.cpu_count() - 2)
    t_map_start=time.perf_counter()
    tokenized = train_data.map(
        pre_process,
        num_proc=CPU,
        batched=True,
        batch_size=1000,
        remove_columns=train_data.column_names,
    )
    print(f"map() took {time.perf_counter()-t_map_start:.3f}s")

    index_queue=Queue()
    index_queue.put(0)
    token_queue=Queue()
    voc=set()
    word_counter=Counter()
    print("token ok")
    work=[]
    print("process start")

    for _ in range(CPU):
        puter=Process(target=queue_put,args=(tokenized["words"],tokenized["counts"],index_queue,token_queue))
        work.append(puter)
        puter.start()
    wall_start=time.perf_counter()
    get_time=0.0
    n_chunks=0
    none_times=0
    voc = set()
    word_counter=Counter()
    while True:
        t0=time.perf_counter()
        data=token_queue.get()
        get_time+=time.perf_counter()-t0
        if data is None:
            none_times+=1
            if none_times==CPU:
                break
            continue
        n_chunks+=1
        chunk_voc,chunk_counter=data
        voc.update(chunk_voc)
        word_counter.update(chunk_counter)
    tuple_counter=Counter()
    for k,v in word_counter.items():
        tuple_counter[tuple(k)]=v
    wall=time.perf_counter()-wall_start
    print(f"[consumer {current_process().pid}] chunks={n_chunks} wall={wall:.3f}s "
          f"get_time={get_time:.3f}s ({100*get_time/wall:.1f}% blocked on IPC)")
    word_counter=tuple_counter

    

# TODO  BPE-merge and hash table and tranformer              and MGNC-CNN
                    # or maybe use open source vec-dataset