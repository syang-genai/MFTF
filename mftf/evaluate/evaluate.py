import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
clip= CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

def lpips_metrics(images):
    eval_images=torch.tensor(images,dtype=torch.float).permute(0,3,1,2)
    lpips_score=lpips(eval_images[[0]]*255*2-1, eval_images[[1]]*255*2-1)
    return lpips_score


def clip_metrics(images,prompts):
    eval_images=torch.tensor(images).permute(0,3,1,2)
    clip_score=clip(eval_images[0],prompts[0]), clip(eval_images[1],prompts[1])
    return clip_score