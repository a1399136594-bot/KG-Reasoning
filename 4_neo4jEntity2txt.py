import csv

# neo4j导出的CSV文件路径
csv_file_path = 'export (8).csv'

# 创建一个空列表来存储数据
data = []

# 使用csv模块的reader函数读取CSV文件
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    # 读取列名（CSV文件的第一行是列名）
    headers = next(csv_reader)
    headers = ["name"]
    # 遍历CSV文件的每一行（从第二行开始，因为第一行是列名）
    for row in csv_reader:
        # 将每一行数据转换为一个字典，其中键是列名
        row_dict = {header: value for header, value in zip(headers, row)}

        # 将字典添加到列表中
        data.append(row_dict)

    # 打印数据以查看结果

file_path1 = "./entity2id.txt"
i = 0
with open(file_path1, 'w', encoding='utf-8') as file:
    for item in data:
        if(len(str(item["name"]))<100): # 过滤长文本节点
            file.write(str(item["name"].strip().split("\"")[1])+'\t'+str(i)+'\n')
            i = i + 1