import re
import json
import asyncio
import websockets
from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel
import json
from neo4j import GraphDatabase

# 连接到Neo4j数据库
uri = "bolt://localhost:7687"
username = "neo4j"
password = "a87797875"
driver = GraphDatabase.driver(uri, auth=(username, password))
Filter_Flag = True

def read_and_chunk_file(file_path, chunk_size=200):
    # chunks = []
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     while True:
    #         chunk = file.read(chunk_size)
    #         if not chunk:
    #             break
    #         chunks.append(chunk)
    # return chunks
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # 按照空行进行分割，每个段落之间会有一个空行
        paragraphs = content.split('\n\n')
    return paragraphs

def parse_json_objects(json_string):
    json_objects = []
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    matches = pattern.findall(json_string)
    for match in matches:
        json_objects.append(json.loads(match))
    return json_objects


# 分类 example
class_examples = {
        '人物': '秦始皇（259BC – 210BC），又称嬴政，是中国历史上著名的政治家和军事家。他是秦国的君主，统一了六国之后建立了中国的第一个中央集权制度的封建王朝——秦朝。',
        '都市': '郑州市隶属于中国河南省，是中华人民共和国的一个地级市。全名为“郑州市”，又称“中原之都”。郑州市是河南省的省会城市，成立省会时间为1952年。截至2021年统计数据，郑州市的总面积为7,446.2平方公里，人口约为1423万。郑州市是河南省的政治、文化、经济中心，也是中国中部地区的重要城市。',
        '装备工艺':'''5.2  分解
5.2.1  防尘罩拆卸
a) 修理人员用两用扳手（13mm）逆时针方向拆下左防尘罩上4颗固定螺栓，将螺栓和弹簧垫圈放入指定位置；（如图3所示）
b) 修理人员取下左防尘罩，放入指定位置；
c) 修理人员用两用扳手（13mm）逆时针方向拆下右防尘罩上4颗固定螺栓，将螺栓和弹簧垫圈放入指定位置；
d) 修理人员取下右防尘罩，放入指定位置。
① 工具：两用扳手（13mm）1把；
② 主材：无；
③ 辅材：无；
④ 耗材：无；
⑤ 人员：修理人员1名；
⑥ 作业时间：0.15h；
⑦ 工时：0.15h。
'''
    }
class_list = list(class_examples.keys())

CLS_PATTERN = f"“{{}}”是 {class_list} 里的什么类别？"


# 定义不同实体下的具备属性
schema = {
    '人物': ['姓名', '性别', '出生日期', '出生地点', '职业', '国籍'],
    '都市': ['别名', '名字', '归属地', '确立时间', '人口'],
    '装备工艺':['名称','工艺流程','工具','主材','耗材','辅材','人员','作业时间','工时']
}

IE_PATTERN = "{}\n\n提取上述句子中{}类型的实体，并按照JSON格式输出，如果有多个实体请输出多个json【即你需要保证输出的名称里面只有一个值！】，上述句子中不存在的信息用【原文中未提及】来表示。请你记住，一个json中的所有属性只能有一个值！！！"
IE_PATTERN_NO_FILTER = "{}\n\n提取上述句子中所有类型的实体，并按照JSON格式输出，多个值之间用','分隔。"

# 提供一些例子供模型参考
ie_examples = {
        '人物': [
                    {
                        'content': '秦始皇（259BC – 210BC），又称嬴政，是中国历史上著名的政治家和军事家。他是秦国的君主，统一了六国之后建立了中国的第一个中央集权制度的封建王朝——秦朝。',
                        'answers': {
                                        '姓名': ['秦始皇'],
                                        '出生日期': ['259BC – 210BC'],
                                        '职业': ['政治家、军事家'],
                                        '功绩': ['统一了六国']
                            }
                    }
        ],
        '都市': [
                    {
                        'content': '郑州市隶属于中国河南省，是中华人民共和国的一个地级市。全名为“郑州市”，又称“中原之都”。郑州市是河南省的省会城市，成立省会时间为1952年。截至2021年统计数据，郑州市的总面积为7,446.2平方公里，人口约为1423万。郑州市是河南省的政治、文化、经济中心，也是中国中部地区的重要城市。南京隶属于江苏，是一个美丽城市，面积200平方，人口只有200人。',
                        'answers': [
                            {
                                        '名字': ['郑州市'],
                                        '别名': ['中原之都'],
                                        '归属地': ['河南省'],
                                        '确立时间': ['1952年'],
                                        '人口': ['1423万']
                            },
                            {
                                '名字': ['南京'],
                                '归属地': ['江苏'],
                                '人口': ['200']
                            }
                        ]
                    }
        ]
}


def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """

    class_list = list(class_examples.keys())
    cls_pre_history = [{'role': 'user',
      'content':  f'现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：{class_list}类别中。'},
     {'role': 'assistant', 'metadata': '', 'content': f'好的。'}
    ]



    for _type, exmpale in class_examples.items():
        cls_pre_history.append({'role': 'user', 'content': f"{exmpale}”是 {class_list} 里的什么类别？请从其中选择一个或者多个你认为的最恰当的类别【用空格隔开】，记住只能在给出的范围内选择！"})
        cls_pre_history.append({'role': 'assistant', 'metadata': '', 'content': f'{_type}'})
    # cls_pre_history.append({'role': 'user','content': f"【动能打击卫星变轨**：红方动能打击卫星从稍低的轨道（约35000公里）逐步调整轨道，接近地球静止轨道上的蓝方卫星。通过一系列精确的轨道调整，动能打击卫星进入蓝方卫星的攻击位置，准备发射动能武器。】是{class_list}里的什么类别？请从其中选择一个或者多个你认为的最恰当的类别【用空格隔开】，记住只能在给出的范围内选择！"})
    # cls_pre_history.append({'role': 'assistant', 'metadata': '', 'content': f'卫星 方案'})
    ie_pre_history = [{'role': 'user',
      'content': "现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中三元组，并按照JSON的格式输出，如果有多个实体请输出多个json【即需要保证名称中只有一个元素，如果有别的，另外输出一个json】， 上述句子中没有的信息用【原文中未提及】来表示，其中所有属性只能有一个值！"},
     {'role': 'assistant', 'metadata': '', 'content': f'好的，请输入您的句子。'}
     ]

    for _type, example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            properties_str = ', '.join(schema[_type])
            schema_str_list = f'“{_type}”({properties_str})'
            sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)
            # ie_pre_history.append((
            #     f'{sentence_with_prompt}',
            #     f"{json.dumps(example['answers'], ensure_ascii=False)}"
            # ))
            ie_pre_history.append({'role': 'user','content': f'{sentence_with_prompt}'})
            if isinstance(example["answers"], list):
                res = ""
                for ans in example["answers"]:
                    res += f'{json.dumps(ans, ensure_ascii=False)}'
                ie_pre_history.append({'role': 'assistant', 'metadata': '',
                                       'content': res})
            else:
                ie_pre_history.append({'role': 'assistant',  'metadata': '','content': f'{json.dumps(example["answers"], ensure_ascii=False)}'})
    return {'ie_pre_history': ie_pre_history, 'cls_pre_history': cls_pre_history}


def clean_response(response: str):
    """
    后处理模型输出。

    Args:
        response (str): _description_
    """
    response = response.replace('、', ',')
    response = response.replace('\'', '"')
    if '```json' in response:
        # res = re.findall(r'```json(.*?)```', response)
        res = response.split("json")[1].split("`")[0]
        # res_sp = []
        # for str_res in res:
        #     if '{' in str_res:
        #         res_sp.append(str_res)
        # if len(res) and res[0]:
        #     response = res[0]
        # response.replace('、', ',')
        return res

    # try:
    #
    #     return json.loads(response)
    #
    # except:
    return response


def inference(
        sentences: list,
        custom_settings: dict
    ):
    """
    推理函数。

    Args:
        sentences (List[str]): 待抽取的句子。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """
    for sentence in sentences:
        with console.status("[bold bright_green] Model Inference..."):
            if Filter_Flag:
                sentence_with_cls_prompt = CLS_PATTERN.format(sentence)
                # cls_res, _ = model.chat(tokenizer, sentence_with_cls_prompt, history=custom_settings['cls_pre_history'])

                custom_settings['cls_pre_history'].append({'role': 'user','content': sentence_with_cls_prompt })
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=custom_settings['cls_pre_history']
                )
                cls_res = str(completion.choices[0].message.content)
                custom_settings['cls_pre_history'].pop()

                for cls_res_1 in class_list:
                    if cls_res_1 not in cls_res:
                        continue
                    if cls_res_1 not in schema:
                        print(f'The type model inferenced {cls_res_1} which is not in schema dict, exited.')
                        exit()
                    properties_str = ', '.join(schema[cls_res_1])
                    schema_str_list = f'“{cls_res_1}”({properties_str})'
                    sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)


                    # ie_res, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=custom_settings['ie_pre_history'])

                    custom_settings['ie_pre_history'].append({'role': 'user','content': sentence_with_ie_prompt })
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=custom_settings['ie_pre_history']
                    )
                    ie_res = str(completion.choices[0].message.content)


                    custom_settings['ie_pre_history'].pop()


                    ie_res = clean_response(ie_res)
                    json_objects = parse_json_objects(ie_res)
                    import_data(json_objects,sentence)
                    print(f'>>> [bold bright_red]sentence: {sentence}')
                    print(f'>>> [bold bright_green]inference answer: ')
                    print(ie_res)


def inference_web(
        sentence: str,
        custom_settings: dict
    ):
    """
    推理函数。

    Args:
        sentences (List[str]): 待抽取的句子。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """

    with console.status("[bold bright_green] Model Inference..."):
        sentence_with_cls_prompt = CLS_PATTERN.format(sentence)
        # cls_res, _ = model.chat(tokenizer, sentence_with_cls_prompt, history=custom_settings['cls_pre_history'])
        custom_settings['ie_pre_history'].append(sentence_with_cls_prompt)
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=custom_settings['cls_pre_history']
        )
        cls_res = str(completion.choices[0].message.content)

        if cls_res not in schema:
            print(f'The type model inferenced {cls_res} which is not in schema dict, exited.')
            exit()

        properties_str = ', '.join(schema[cls_res])
        schema_str_list = f'“{cls_res}”({properties_str})'
        sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)
        # ie_res, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=custom_settings['ie_pre_history'])

        custom_settings['ie_pre_history'].append(sentence_with_ie_prompt)
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=custom_settings['ie_pre_history']
        )
        ie_res = str(completion.choices[0].message.content)


        ie_res = clean_response(ie_res)
    print(f'>>> [bold bright_red]sentence: {sentence}')
    print(f'>>> [bold bright_green]inference answer: ')
    print(ie_res)
    return ie_res


def create_node(tx, label, properties):
    query = (
            f"CREATE (n:entity{{"
            + ", ".join(f"{k}: ${k}" for k in properties.keys())
            + "}) RETURN id(n) as id"
    )
    result = tx.run(query, **properties)
    return result.single()["id"]


def create_relationship(tx, start_id, end_id, rel_type):
    query = (
            "MATCH (a), (b) WHERE ID(a) = $start_id AND ID(b) = $end_id "
            "CREATE (a)-[r:" + rel_type + "]->(b) RETURN r"
    )
    tx.run(query, start_id=start_id, end_id=end_id)


def import_data(json_objects,sentence_with_ie_prompt):
    with driver.session() as session:
        name_properties = {"value": sentence_with_ie_prompt}
        text_node_id = session.write_transaction(create_node, "text", name_properties)
        shouce_node_id = session.write_transaction(create_node, "text",  {"value": "轮毂工艺手册"})
        session.write_transaction(create_relationship, shouce_node_id, text_node_id, "涉及")
        for json_data in json_objects:
            name_nodes = json_data.get("名称", [])
            if not isinstance(name_nodes, list):
                name_nodes = [name_nodes]
            for name in name_nodes:
                name_properties = {"value": name}
                name_node_id = session.write_transaction(create_node, "名称", name_properties)
                session.write_transaction(create_relationship, text_node_id, name_node_id, "相关")
                for key, values in json_data.items():
                    if key == "名称":
                        continue
                    if not isinstance(values, list):
                        values = [values]
                    for value in values:
                        if "未提及" in value:
                            continue
                        properties = {"value": value}
                        node_id = session.write_transaction(create_node, key, properties)
                        session.write_transaction(create_relationship, name_node_id, node_id, key)


from openai import OpenAI
import httpx

console = Console()
# device = 'cuda:0'
# tokenizer = AutoTokenizer.from_pretrained("C:\\Users\win11\\dataroot\\models\\THUDM\\chatglm3-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("C:\\Users\win11\\dataroot\\models\\THUDM\\chatglm3-6b", trust_remote_code=True).half().quantize(4)
# model.to(device)

client = OpenAI(
    base_url="https://api.xty.app/v1",
    api_key="sk-9bAadClSFpQfbYqq1f8eCcBaF3B14254A6489f93F74635C0",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

custom_settings = init_prompts()


# 使用示例
file_path = 'mini.txt'  # 请将此处替换为你的txt文件路径
chunks = read_and_chunk_file(file_path)
inference(
        chunks,
        custom_settings
    )
# 输出结果

