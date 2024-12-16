import torch
from diffusers import StableDiffusionPipeline

from diffusion import text2image_ldm_stable
from attention import MutualSelfAttentionControlMaskAuto

from utils import prompts_idx, view_images, lpips_metrics, clip_metrics

g_cpu = torch.Generator().manual_seed(42)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = ldm_stable.tokenizer


GUIDANCE_SCALE=7.5
NUM_DIFFUSION_STEPS=30

prompts = [
    "a crystal-clear stream winds through the forest.", 
    "a crystal-clear stream winds through the forest.", 
]

prompts_ids =prompts_idx(prompts,tokenizer)
print("prompt_ids",prompts_ids)

layer_idx=list(range(0,16))
step_idx=list(range(0,11))

ref_token_idx=[prompts_ids[0]["a"]+prompts_ids[0]["crystal-clear"]+prompts_ids[0]["stream"]]
thres=[0.2]
object_afflines=[{"angle":0,"translate":(0.2,0.0),"scale":1.0,"shear":0}]

editor = MFTF(start_step=0, start_layer=0, layer_idx=layer_idx, step_idx=step_idx, total_steps=NUM_DIFFUSION_STEPS, thres=thres, ref_token_idx=ref_token_idx, object_afflines=object_afflines, mask_save_dir='/working', model_type="SDXL")
images, x_t = text2image_ldm_stable(ldm_stable, prompts, editor, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, latent=None, low_resource=False,generator=g_cpu)

selected_prompts=[["a crystal-clear stream"]]
view_images(images,prompts,selected_prompts,object_afflines,fixed_width=70,fontsize=11,save_image_path='/working/stream_desert.png')
print("lpips",lpips_metrics(images))
print("clip",clip_metrics(images)) 