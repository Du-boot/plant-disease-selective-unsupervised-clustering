import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.font_manager as fm


# font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # 替换为你的字体路径
# font_prop = fm.FontProperties(fname=font_path)
# print(font_prop.get_name())  # 输出字体的真实名称
# exit()

# 获取所有图片的实际标签和预测标签
def get_labels(root_dir):
    y_true = []
    y_pred = []
    for pred_label in os.listdir(root_dir):
        pred_label_path = os.path.join(root_dir, pred_label)
        if pred_label == 'rest':  # 跳过名为 'rest' 的子文件夹
            continue
        if os.path.isdir(pred_label_path):
            for image_file in os.listdir(pred_label_path):
                if re.match(r'^\d+_\d+\.jpg$', image_file):  #jpg
                    true_label = int(image_file.split('_')[0])
                    y_true.append(true_label)
                    y_pred.append(int(pred_label))
    if not y_true or not y_pred:
        raise ValueError("No valid labels found. Please check the directory structure and file naming conventions.")
    return y_true, y_pred
# def get_labels(root_dir):
    # """
    # 从目录结构中获取真实标签和预测标签。
    # - 预测标签：子文件夹名里的数字（比如 '0', '1', 'cluster_2' 等）
    # - 真实标签：图片文件名最前面的数字，比如
      # '13-02e76b75-f201-44ee-a694-35edf97cc82b___R.S_HL 8015 copy.jpg' -> 13
    # """
    # y_true = []
    # y_pred = []

    # for pred_label in os.listdir(root_dir):
        # if pred_label == 'rest':  # 跳过 rest
            # continue

        # pred_label_path = os.path.join(root_dir, pred_label)
        # if not os.path.isdir(pred_label_path):
            # continue

        # 从文件夹名里抽出数字作为“预测标签”
        # m_dir = re.search(r'\d+', pred_label)
        # if not m_dir:
            # 这个文件夹名里根本没数字，跳过
            # continue
        # pred_label_int = int(m_dir.group())

        # for image_file in os.listdir(pred_label_path):
            # 只看常见图片格式
            # if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # continue

            # 从文件名最前面解析真实标签： ^(\d+)- ...
            # m_file = re.match(r'^(\d+)-', image_file)
            # if not m_file:
                # 文件名不符合“数字-开头”的就跳过
                # continue

            # true_label = int(m_file.group(1))

            # y_true.append(true_label)
            # y_pred.append(pred_label_int)

    # if not y_true or not y_pred:
        # raise ValueError(
            # "No valid labels found. Please check the directory structure and file naming conventions."
        # )
    # return y_true, y_pred

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels, label_names,  font_size=17, label_size = 20,label_pad=20):  #font_size=12, label_size = 15,label_pad=20  font_size=17, label_size = 20,label_pad=20

    #*
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 使用 Noto Sans CJK
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 使用文泉驿正黑
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # plt.rcParams['font.family'] = font_prop.get_name()
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    #*
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, annot_kws={"size": font_size})
    plt.xlabel('预测类别', fontsize=label_size, labelpad=label_pad)
    plt.ylabel('真实类别', fontsize=label_size, labelpad=label_pad)
    plt.xticks(fontsize=font_size,rotation=45, ha='right')
    plt.yticks(fontsize=font_size, rotation=45, va='center')
    plt.title('Wuhan East Lake', fontsize=label_size) #Wuhan East Lake  洱海
    plt.show()

# 主函数
if __name__ == "__main__":
    root_dir = '/data1/D/deploy/20250929_129/merged_result_final20'  #1_result
    
    y_true, y_pred = get_labels(root_dir)
    labels = sorted(set(y_true + y_pred))  # 获取所有标签

    # 定义标签名称，例如 {1: "xx物种", 2: "yy物种"}
    # label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}  # 根据需要修改
    # label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}  # 根据需要修改
    # label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}  # 根据需要修改
    
    
    #中文
    # label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}  # 根据需要修改
    label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}  # 根据需要修改
    # label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}
    # label_names_dict = {0: "Ferny", 1: "Rounded", 2:"Strappy", 3:"Substrate"}  # 根据需要修改
    
    
    label_names = [label_names_dict.get(label, str(label)) for label in labels]

    if not labels:
        raise ValueError("No valid labels found after processing the files. Please ensure the directory structure and file naming are correct.")
    
    plot_confusion_matrix(y_true, y_pred, labels, label_names)
    print(classification_report(y_true, y_pred, target_names=label_names))
