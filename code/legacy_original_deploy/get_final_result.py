import config as CF
import glob
import os,shutil

def get_first_label(file_path):
    files=glob.glob(file_path+"/*")
    ret_im2label={}
    ret_label2im={}
    for files_ in files:
        label=files_.split("/")[-1]
        if label not in ret_label2im:
            ret_label2im[label]=[]
        ims_path=glob.glob(files_+"/*")
        for ims in ims_path:
            ims=ims.split("/")[-1]
            ret_im2label[ims]=label
            ret_label2im[label].append(ims)
    return ret_im2label,ret_label2im
def get_max(t):
    max_=""
    max_v=0
    for key,value in t.items():
        if value >max_v:
            max_=key
            max_v=value
    return max_

def get_non_first_label(t_first_im2label,file_path):
    files=glob.glob(file_path+"/*")
    ret_label2im={}
    #print("in non first,%s"%len(files)) 
    #print(files)
    # print(files)
    # print("?????????????????")
    for files_ in files:
        t_count={}
        tmp=[]
        ims_path=glob.glob(files_+"/*")
        for ims in ims_path:
            # print(ims)
            ims=ims.split("/")[-1]
            label=t_first_im2label[ims]
            t_count[label]=t_count.get(label,0)+1
            tmp.append(ims)
        # print(t_count)
        this_label=get_max(t_count)
        # print(this_label)
        ret_label2im[this_label]=tmp
        tmp=[]
    return ret_label2im

def keep(file_path,class_num):
    files=glob.glob(file_path+"/*")
    if len(files)>0 and len(files)<=class_num:
        return True
    else:
        return False
def make_blank_dict(raw_path):
    t={}
    raw_file_lst=glob.glob(raw_path+"/*")
    for raw in raw_file_lst:
        t[raw]={}
    # print("there are %s clusters"%len(t))
    return t
def inter(lst1,lst2):
#    print(lst1,lst2)
    ret=[]
    for x in lst1:
        if x in lst2:
            ret.append(x)
    #ret=[v for v in lst1 if v in lst2]
    return ret

def get_final(class_num):
    #class_num=CF.config["class_num"]
    # class_num = int(input('请输入数据类别总数：'))
    
    folder = "1_new"
    # folder = "2_new"
    
    class_num = int(class_num)
    t_whole={}
    files=glob.glob("20250929_129/result/*")
    # print(files)
    # exit()
    for fi in files:
        # print(len(glob.glob("%s/*"%fi)),glob.glob("%s/*"%fi),class_num)
        # exit()
        # if fi.find("k")==-1:continue
        # else:
            # break
        #print(glob.glob("%s/*"%fi))
        if len(glob.glob("%s/*"%fi))==class_num:  
            # print("the first chosen file is %s"%fi) 
            t_im2label_first,t_label2im_first=get_first_label(fi)
            for label,im in t_label2im_first.items():
                pass
                # print(label,len(im))
            #print(t_label2im_first)
          #  exit()
            files.remove(fi)
            t_whole[fi]=t_label2im_first 
            break
        elif not keep(fi,class_num):
            files.remove(fi)
    # print(t_im2label_first,'==t_im2label_first=')
    #exit()
    for fi in files:
        if keep(fi,class_num):
            # print(t_im2label_first,'==t_im2label_first=')
            t_label2im_=get_non_first_label(t_im2label_first,fi)
            # for label,im in t_label2im_.items():
                # print(label,len(im))
            #exit()
            t_whole[fi]=t_label2im_
            # print(len(t_label2im_),fi)
        else:
            pass
            # print("abandom %si"%(fi))
    #exit()

    t_final=t_label2im_first
    # print("一共有%s个聚类方法"%(len(t_whole)))
    for fi,items in t_whole.items():
        # print(len(items))
        for label,ims in items.items():
            if label not in t_final:
                t_final[label]=[]
            # print("现在处理聚类方法%s。取并集前，类别%s 有%s个样本"%(fi,label,len(t_final[label])))
            t_final[label]=inter(t_final[label],ims)
            # print("取并集后，类别%s 有%s个样本"%(label,len(t_final[label])))
    result_final_dir = "20250929_129/result_final%s" % class_num
    if os.path.isdir(result_final_dir):
        shutil.rmtree(result_final_dir)
    os.makedirs(result_final_dir)
    ims_all=[x.split("/")[-1] for x in glob.glob(r"../%s/*"%folder)]

    # print(ims_all)
    # print(glob.glob('..\\*'))
    # print(glob.glob(r"..\Data\raw\*"))
    # exit()
    total_num_pic=len(ims_all)
    for label,ims in t_final.items():
        # print(ims)
        path_t="20250929_129/result_final%s/%s"%(class_num,label)
        if glob.glob(path_t)==[]:
            #os.system("mkdir %s"%path_t)
            os.makedirs("%s"%path_t)
        else:
            # os.system("rm -r %s && mkdir %s"%(path_t,path_t))
            shutil.rmtree("%s"%(path_t))
            os.makedirs("%s"%(path_t))
        for im in ims:
            # print(ims_all)
            # print(im)
            ims_all.remove(im)
            # os.system("cp pic/%s %s"%(im,path_t))
            shutil.copy(r"../%s/%s"%(folder,(im)),'%s'%path_t)
        # print(label)
        #print(ims)
        # print("===============")
    # print("total number is %s, the number of well classified is %s"%(total_num_pic,len(ims_all)))
    rest_file_path="20250929_129/result_final%s/rest"%class_num
    if glob.glob(rest_file_path)!=[]:
        # os.system("rm -r %s"%rest_file_path)
        shutil.rmtree("%s"%(rest_file_path))
    # os.system("mkdir %s"%rest_file_path)
    os.makedirs("%s"%rest_file_path)
    for im in ims_all:
        # os.system("cp pic/%s %s"%(im,rest_file_path))
        # shutil.copy(r"../Data/%s/%s"%(folder,(im)),'%s'%rest_file_path)
        shutil.copy(r"../%s/%s"%(folder,(im)),'%s'%rest_file_path)
  

if __name__ == '__main__':
    
    get_final(2)
        
# get_final_result.py —— 多算法“多数票(>=2/3)”投票融合；支持 K>3；落地 result_final{K}/
# import os, shutil, collections

# 可配置：多数票阈值（2 表示至少 2/3 同意）
# MIN_VOTES = 2

# def _list_dirs(p):
    # return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])

# def _list_imgs(p):
    # imgs = []
    # for fn in os.listdir(p):
        # if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            # imgs.append(fn)
    # return imgs

# def _get_true_label_from_name(name):
    # 取文件名前缀的整数部分，如 2_123.jpg -> 2
    # try:
        # return int(os.path.basename(name).split("_", 1)[0])
    # except Exception:
        # 退路：提取开头连续数字
        # import re
        # m = re.match(r"(\d+)", os.path.basename(name))
        # return int(m.group(1)) if m else -1

# def _scan_algo(algo_root):
    # """读取某个算法的结果树：{cluster_label(str): set(image_name)}"""
    # m = {}
    # if not os.path.isdir(algo_root): 
        # return m
    # for lab in _list_dirs(algo_root):
        # p = os.path.join(algo_root, lab)
        # names = set(_list_imgs(p))
        # if names:
            # m[lab] = names
    # return m

# def _find_anchor(result_root, K, algos):
    # """优先选择“簇数==K”的算法作为锚；否则选簇数<=K且最大的"""
    # anchor_algo, anchor_map = None, None
    # best = (-1, None, None)
    # for a in algos:
        # mp = _scan_algo(os.path.join(result_root, a))
        # k = len(mp)
        # if k == K:
            # return a, mp
        # if 0 < k <= K and k > best[0]:
            # best = (k, a, mp)
    # return best[1], best[2]

# def get_final(class_num, result_root="result", out_prefix="result_final"):
    # """
    # 多算法多数票融合：
    # 1) 选一个算法为锚（优先簇数==K），用其目录名当“锚标签”；
    # 2) 其它算法按锚标签对齐：每个簇映射到出现在该簇样本里的“锚标签众数”；
    # 3) 对每张图，统计各算法投的“锚标签”，>=MIN_VOTES 归入该类，否则进 rest。
    # """
    # K = int(class_num)
    # algos = [d for d in _list_dirs(result_root) if d not in (out_prefix,)]
    # if not algos:
        # raise RuntimeError(f"未找到算法结果目录：{result_root}/*")

    # anchor_algo, anchor_mp = _find_anchor(result_root, K, algos)
    # if not anchor_algo:
        # raise RuntimeError("没有找到可用的锚算法（簇数<=K）。请先运行 main.cluster_raw(...)。")
    # print(f"[融合] 参与算法：{algos}；锚算法={anchor_algo}（簇数={len(anchor_mp)}）")

    # 1) 先收集全部样本名
    # all_names = set()
    # for a in algos:
        # mp = _scan_algo(os.path.join(result_root, a))
        # for s in mp.values():
            # all_names.update(s)
    # if not all_names:
        # raise RuntimeError("未在 result/*/* 中找到图片。")

    # 2) 锚算法的 name->label 映射（锚标签）
    # anchor_of = {}
    # for lab, names in anchor_mp.items():
        # for n in names:
            # anchor_of[n] = lab

    # 3) 开始计票：先给锚算法计一票
    # votes = {n: collections.Counter() for n in all_names}
    # for n, lab in anchor_of.items():
        # votes[n][lab] += 1

    # 4) 其它算法 -> 对齐到锚标签，再计票
    # for a in algos:
        # if a == anchor_algo: 
            # continue
        # mp = _scan_algo(os.path.join(result_root, a))
        # 对齐：该算法的每个簇，取其中样本的“锚标签众数”作为映射目标
        # mapping = {}  # 本算法目录名 -> 锚标签
        # for lab, names in mp.items():
            # cnt = collections.Counter(anchor_of.get(n, None) for n in names)
            # cnt.pop(None, None)
            # if not cnt:
                # continue
            # anchor_lab, _ = cnt.most_common(1)[0]
            # mapping[lab] = anchor_lab

        # 计票
        # for lab, names in mp.items():
            # anchor_lab = mapping.get(lab, None)
            # if anchor_lab is None:
                # continue
            # for n in names:
                # votes[n][anchor_lab] += 1

    # 5) 多数票决策
    # out_dir = f"{out_prefix}{K}"
    # if os.path.isdir(out_dir):
        # shutil.rmtree(out_dir)
    # os.makedirs(out_dir, exist_ok=True)
    # for i in range(K):
        # os.makedirs(os.path.join(out_dir, str(i)), exist_ok=True)
    # os.makedirs(os.path.join(out_dir, "rest"), exist_ok=True)

    # kept, dropped = 0, 0
    # for n, c in votes.items():
        # if not c:
            # dst = os.path.join(out_dir, "rest", n)
        # else:
            # pred, v = c.most_common(1)[0]
            # if v >= MIN_VOTES:
                # dst = os.path.join(out_dir, str(pred), n)
                # kept += 1
            # else:
                # dst = os.path.join(out_dir, "rest", n)
                # dropped += 1
        # 源图默认在 ../1_new/
        # src = os.path.join("..", "1_new", n)
        # try:
            # shutil.copy(src, dst)
        # except Exception as e:
            # 找不到就跳过，但仍然统计决策
            # print(f"[WARN] 拷贝失败 {src} -> {dst}: {e}")

    # total = kept + dropped
    # print(f"[融合] 多数票阈值={MIN_VOTES}；保留={kept} / {total} ({kept/total:.2%})，丢弃={dropped/total:.2%}")
    # return kept/total, dropped/total
