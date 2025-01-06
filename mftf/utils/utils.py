import os
import torch
import collections 
import matplotlib.pyplot as plt
from textwrap import wrap

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore

def prompts_idx(prompts,tokenizer):
    word_ids=[]
    for sent in prompts:
        idx=0
        temp=[idx]
        for w in sent.split():
            temp=temp+[idx+1 for i in range(len(tokenizer.tokenize(w)))]
            idx=idx+1
        temp.append(idx+1)
        word_ids.append(temp)

    prompts_ids=[]
    for idxs, words in zip(word_ids,prompts):
        words_idxs=collections.defaultdict(list)
        words_list=words.split()
        for i in range(1,len(idxs)-1):
            words_idxs[words_list[idxs[i]-1]].append(i)
        prompts_ids.append(words_idxs)
    return prompts_ids


def view_images(images,prompts,selected_prompts,object_afflines,fixed_width,fontsize=12,save_image_path="."):
    def flatten_dict_to_string(d):
        return ", ".join(f"{key}:{value}" for key, value in d.items())
    
    plt.figure(figsize=(5*images.shape[0],5))
    for i in range(images.shape[0]):
        plt.subplot(1,images.shape[0],i+1)
        plt.imshow(images[i])
        if i==1:
            title=""
            for j, oa in enumerate(object_afflines):
                if "shear" in oa:
                    oa.pop("shear")
                title=title+"\n"+" ".join(selected_prompts[j])+": "+flatten_dict_to_string(oa)
            wrapped_caption=title+"\n"+"prompt:"+" "+"\n".join(wrap(prompts[i],fixed_width))
            plt.title(wrapped_caption,fontsize=fontsize)
        else:
            title=prompts[i]
            wrapped_caption="\n".join(wrap(title, fixed_width))
            plt.title("prompt:"+" "+wrapped_caption,fontsize=fontsize)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(save_image_path)
    return