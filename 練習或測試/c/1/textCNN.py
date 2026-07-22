from datasets import load_dataset
import os
from collections import Counter
from multiprocessing import Process,Queue

def pre_process(data):
    return {"tokens": [sentence.split(" ") for sentence in data["text"]]}

def queue_put(iter,idx_queue,queue):
    idx=-1
    while True:
        idx=idx_queue.get()
        if idx>=len(iter):
            idx_queue.put(idx)
            break
        queue.put(iter[idx])
        idx_queue.put(idx+1)
        
        
def text_count(queue,return_queue):
    voc = set()
    word_counter=Counter()
    while True:
        data=queue.get()
        if data is None:
            queue.put(None)
            break
        voc.update("".join(data))
        word_counter.update(data)
    return_queue.put((voc,word_counter))


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
    tokenized = train_data.map(
        pre_process,
        num_proc=CPU,
        batched=True,
        batch_size=1000,
        remove_columns=train_data.column_names,
    )
    index_queue=Queue()
    index_queue.put(0)
    token_queue=Queue()
    result_queue=Queue()
    voc=set()
    word_counter=Counter()
    print("token ok")
    work=[]
    print("process start")

    for _ in range(CPU//2):
        puter=Process(target=queue_put,args=(tokenized["tokens"],index_queue,token_queue))
        counter=Process(target=text_count,args=(token_queue,result_queue))
        work.append(puter)
        work.append(counter)
        puter.start()
        counter.start()
    for puter in range(0,len(work),2):
        work[puter].join()
    print("puter done")
    token_queue.put(None)

    num_consumers = CPU // 2
    for _ in range(num_consumers):
        v,w=result_queue.get()
        voc.update(v)
        word_counter.update(w)
    for worker in work:
        worker.join()
    voc.update(" ")
    print("process end",voc.pop(),voc.pop(),word_counter["I"],word_counter["a"])
# TODO better multiprocessing code, BPE-merge and MGNC-CNN