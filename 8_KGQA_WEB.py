import asyncio
import websockets
import re
import threading

from neo4j import GraphDatabase, basic_auth
import pickle
from openai import OpenAI
import httpx
import pandas as pd
import numpy as np
import itertools
from typing import Dict, List

import tkinter as tk
from tkinter import scrolledtext


# 1. build neo4j knowledge graph datasets
uri = "bolt://localhost:7687"
username = "neo4j"
password = "a87797875"

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()


client = OpenAI(
    base_url="https://api.xty.app/v1",
    api_key="sk-9bAadClSFpQfbYqq1f8eCcBaF3B14254A6489f93F74635C0",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)
exist_entity = None

re1 = r'所有提取到的实体为(.*?)<EOS>'
re2 = r"提取到的实体是(.*?)<END>"
re3 = r"<CLS>(.*?)<SEP>"


#实体到向量的转化
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# load data
import json, pdb, re

with open("./entity2id.txt","r",encoding="utf-8") as f:
    entities = f.readlines()
    entities = [entity.strip().split()[0] for entity in entities]

keywords = set([])
keywords.add("Headache")
keywords.add("Cough")
# with open("dataset3_ner.json", "r") as f:
#     for line in f.readlines():
#         x = json.loads(line)
#
#         question_kg = x["question_kg"]
#         question_kg = question_kg.replace("\n","")
#         kws = question_kg.split(", ")
#
#         [keywords.add(kw) for kw in kws]
keywords = list(keywords)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./distiluse-base-multilingual-cased-v1')


# encode entities
embeddings = model.encode(entities, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entities,
    "embeddings": embeddings,
}
import pickle
with open("entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)

# encode keywords
embeddings = model.encode(keywords, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": keywords,
    "embeddings": embeddings,
}
import pickle
with open("keyword_embeddings.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("done!")



def final_answer(str1, response_of_KG_list_path):
    messagex = [
        {"role": "user", "content": f"你是一个杰出的助手！你可以帮我分析我所给出的问题！我给出的问题是：{str1}。\n 你有一些知识三元组信息如下所示:\n\n" + '###' + response_of_KG_list_path + '\n\n ##下面请你根据我给出的知识三元组，给出你的回答：【回答简洁明了，不要反问我】'},
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messagex
        )
        strx = str(completion.choices[0].message.content)
    except Exception as e:
        print("An error occurred:", e)
    return strx
def prompt_neighbor(neighbor):
    messagex = [
    ]
    template = f"""
    There are some knowledge graph. They follow entity->relationship->entity list format.
    \n\n
    {neighbor}
    \n\n
    Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...\n\n

    Output:
    """

    try:
        messagex.append({"role": "user", "content": template})
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messagex
        )
        strx = str(completion.choices[0].message.content)
    except Exception as e:
        print("An error occurred:", e)
    question_kg = re.findall(re1, strx)
    return question_kg
def prompt_path_finding(path_input):
    messagex = [
    ]
    Path = path_input
    template = f"""
    下面是一些知识图谱路径， 他们遵循 实体->关系->实体 的格式.
    \n\n
    {Path}
    \n\n
    使用这些知识图谱信息.试着将他们分别转化为自然语言. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...\n\n

    Output:
    """

    try:
        messagex.append({"role": "user", "content": template})
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messagex
        )
        strx = str(completion.choices[0].message.content)
    except Exception as e:
        print("An error occurred:", e)
    question_kg = re.findall(re1, strx)
    return question_kg
def prompt_extract_keyword(input_text):
    messagex = [
        {"role": "system", "content": "你是一个杰出的助手！"},
    ]
    input = input_text
    template = f"""
    帮助我提取实体 ,下面是一些例子:
    \n\n
    ### Instruction:\n'学习从以下问题中提取实体。请你记住，最多提取5个实体'\n\n### Input:\n
    <CLS>发动机中的燃油滤清器发生堵塞，我现在该怎么办?<SEP>所有提取到的实体为\n\n ### Output:
    <CLS>发动机中的燃油滤清器发生堵塞，我现在该怎么办?<SEP>所有提取到的实体为发动机, 燃油滤清器<EOS>
    \n\n
    下面你帮我尝试输出:
    ### Instruction:\n'学习从以下问题中提取实体。'\n\n### Input:\n
    <CLS>{input}<SEP>所有提取到的实体为\n\n ### Output:
    """
    try:
        messagex.append({"role": "user", "content": template})
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=messagex
        )
        strx = str(completion.choices[0].message.content)
    except Exception as e:
        print("An error occurred:", e)
    question_kg = re.findall(re1,strx)
    return question_kg


def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def get_entity_neighbors(entity_name: str, disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]

        if disease_flag == 1 :
            continue

        neighbors = record["neighbor_entities"]

        if "belongTo" in rel_type.replace("_", " "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_", " "), rel_type.replace("_", " "),
                                  ','.join([x.replace("_", " ") for x in neighbors])
                                  ])

    return neighbor_list, disease
def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results
def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..10]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)

            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_", " ")

                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_", " ")
                    path_str += "->" + relations[i] + "->"

            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}

        if len(paths) > 5:
            paths = sorted(paths, key=len)[:5]

        return paths, exist_entity
async def echo_and_send(websocket, path):
    async for message in websocket:
        QA = message
        print('Question:\n', QA)
        await websocket.send(f"{QA}")

        question_kg = prompt_extract_keyword(QA)
        question_kg = question_kg[0].replace("<END>", "").replace("<EOS>", "")
        question_kg = question_kg.replace("\n", "")
        question_kg = question_kg.split(", ")
        if len(question_kg) == 0:
            print("<Warning> no entities found", input)
        print(question_kg)
        keywords = set([])
        for i in question_kg:
            keywords.add(i)
        keywords = list(keywords)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('./distiluse-base-multilingual-cased-v1')
        embeddings = model.encode(keywords, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
        keyword_emb_dict = {
            "keywords": keywords,
            "embeddings": embeddings,
        }

        import pickle

        with open("keyword_embeddings.pkl", "wb") as f:
            pickle.dump(keyword_emb_dict, f)

        # 读取已经预训练好的向量
        with open('./entity_embeddings.pkl', 'rb') as f1:
            entity_embeddings = pickle.load(f1)

        with open('./keyword_embeddings.pkl', 'rb') as f2:
            keyword_embeddings = pickle.load(f2)

        match_kg = []
        entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
        for kg_entity in question_kg:

            keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

            cos_similarities = cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            max_index = cos_similarities.argmax()

            match_kg_i = entity_embeddings["entities"][max_index]
            while match_kg_i.replace(" ", "_") in match_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = entity_embeddings["entities"][max_index]

            match_kg.append(match_kg_i.replace(" ", "_"))
        # print('match_kg',match_kg)

        # # 4. neo4j knowledge graph path finding
        neighbor_list = []
        if len(match_kg) != 0:
            for entity in match_kg:
                query = """
                    MATCH p=(e)-[r*1..2]-(n)
                    WHERE e.value = $entity_name
                    RETURN p
                    """
                result = session.run(query, entity_name=entity)

                for record in result:
                    path = record["p"]
                    # 初始化路径字符串
                    path_str = ""
                    # 遍历路径中的节点和关系
                    for i, node in enumerate(path.nodes):
                        # 添加节点到路径字符串
                        if 'values' in node:
                            path_str += str(node["name"]) + "【"+str(node["values"])+"】"
                        else:
                            # path_str += str(node["name"])
                            path_str += str(node["value"])
                        # 如果不是最后一个节点，添加关系和箭头到路径字符串
                        if i < len(path.nodes) - 1:
                            rel = path.relationships[i]
                            # 添加关系到路径字符串，你可以根据需要格式化关系的表示
                            path_str += " -> " + str(type(rel).__name__) + " -> "  # 或者使用rel.type()获取关系类型
                    # 将路径字符串添加到neighbor_list中
                    neighbor_list.append(path_str)

        list_kg = "\n"
        for pathx in neighbor_list:
            list_kg += pathx + '\n'
        res = final_answer(QA, list_kg)
        print(res)
        await websocket.send(f"answer@{res}<|im_end|>")


start_server = websockets.serve(echo_and_send, "localhost", 8765)



# 运行服务器直到被停止
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()







