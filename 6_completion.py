with open("./entity2id.txt","r",encoding="utf-8") as f:
    entities = f.readlines()
    entities = [entity.strip().split()[0] for entity in entities]

## 将所有实体给到gpt得到文本
prompt = '''下面我将发你一大串neo4j导出的实体，其中有一些实体为部件实体，有一些为故障现象、流程等，我需要你将故障现象和流程等可能对应的实体标注出来，我现在给你一个例子：
【喷油嘴堵塞 -- 喷油嘴】
这个代表喷油嘴堵塞这个现象对应了喷油嘴这个部件'''

from neo4j import GraphDatabase

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "a87797875"))

def create_relationship(tx, head_entity, tail_entities):
    for tail_entity in tail_entities:
        if tail_entity in entities and head_entity in entities:
            tx.run("""
                MATCH (h {value: $head_entity}), (t {value: $tail_entity})
                CREATE (h)-[:相关设备]->(t)
                """, head_entity=head_entity, tail_entity=tail_entity)

def process_file_and_create_relationships(file_path):
    with driver.session() as session:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if '-' in line:
                    head_entity, tail_part = line.split(' - ')
                    head_entity = head_entity.strip()
                    tail_entities = [entity.strip() for entity in tail_part.split('、')]
                    session.write_transaction(create_relationship, head_entity, tail_entities)

if __name__ == "__main__":
    # 指定你的txt文件路径
    file_path = './completion.txt'
    process_file_and_create_relationships(file_path)

# 关闭 Neo4j 连接
driver.close()
