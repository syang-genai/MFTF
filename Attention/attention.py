# code modifications based on the following implementations:
# MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing
# Prompt-to-Prompt Image Editing with Cross Attention Control

from einops import rearrange, repeat
import torch.nn as nn
from torchvision.transforms import functional
from torchvision.utils import save_image

class AttentionBase:
    """
        base class for attention modification
    """
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()
        return out


    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("MFTF at denoising steps: ", self.step_idx)
        print("MFTF at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
            performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
            attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        
        attnu, attnc = attn.chunk(2)
        
        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out


class MFTF(MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=[0.1], ref_token_idx=[1], object_afflines=dict(), mask_save_dir=None, model_type="SD"):
        """
        MFTF with mask auto generation from cross-attention map
        args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation: aggregation for all attention layers
            mask_save_dir: the path to save the mask image
        """
        
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        self.thres = thres
        self.ref_token_idx = ref_token_idx

        self.self_attns = []
        self.cross_attns = []
        
        self.object_afflines=object_afflines
        
        self.cross_attns_mask = None
        self.self_attns_mask_ref= None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)
    

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []


    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
            performing attention for a batch of queries, keys, and values
        """
        
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")

        if self.self_attns_mask_ref is not None:
            self.mask_ref = self.self_attns_mask_ref.reshape(H,W,len(self.ref_token_idx)).unsqueeze(0).unsqueeze(0).clone()
            self.save_masks(self.mask_ref,f"mask_source")
            self.creat_mask()
            
            self.save_masks(self.mask_ref,f"mask_ref_object_source")
            self.save_masks(1-self.mask_ref,f"mask_ref_background_source")

            q_save=q.permute(0,2,1).reshape(B*num_heads,q.shape[-1],H,W)[:1,1,:,:]
            save_image(q_save, os.path.join(self.mask_save_dir, f"q_source_step_{self.cur_step}_layer_{self.cur_att_layer}.png"))
            
            q_fg=q.permute(0,2,1).reshape(B*num_heads,q.shape[-1],H,W)
            q_fg=self.q_affine_mask(q_fg,H,W)
            save_image(q_fg[:1,1,:,:], os.path.join(self.mask_save_dir, f"q_source_fg_affline_step_{self.cur_step}_layer_{self.cur_att_layer}.png"))
            q_fg=q_fg.reshape(B*num_heads,q.shape[-1],H*W).permute(0,2,1)
            
            q_bg=q.permute(0,2,1).reshape(B*num_heads,q.shape[-1],H,W).masked_fill(self.mask_ref.sum(dim=-1).type(torch.bool),0)
            save_image(q_bg[:1,1,:,:], os.path.join(self.mask_save_dir, f"q_source_bg_step_{self.cur_step}_layer_{self.cur_att_layer}.png"))
            q_bg=q_bg.reshape(B*num_heads,q.shape[-1],H*W).permute(0,2,1)
        
            
            q_fg = rearrange(q_fg, "(b h) n d -> h (b n) d", h=num_heads)
            q_bg = rearrange(q_bg, "(b h) n d -> h (b n) d", h=num_heads)

            sim_fg = torch.einsum("h i d, h j d -> h i j", q_fg, k) * kwargs.get("scale")
            sim_bg = torch.einsum("h i d, h j d -> h i j", q_bg, k) * kwargs.get("scale")

            sim = torch.cat([sim_fg, sim_bg])
        
        attn = sim.softmax(-1)
        
        if len(attn) == 2 * len(v):
            v = torch.cat([v,v])

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out


    def aggregate_cross_attn_map(self, idx_list):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, image_seq length, text_seq length)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        
        image=torch.zeros(B,res,res,len(idx_list)).to(attn_map.device)
        for i, idx in enumerate(idx_list):
            image[:,:,:,[i]]=self._aggregate_cross_attn_map(attn_map,idx)
        return image


    def _aggregate_cross_attn_map(self, attn_map,idx):        
        image = attn_map[..., idx].mean(dim=-1,keepdim=True)

        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image


    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
            attention forward function
        """
        if is_cross:
            if attn.shape[1]==256:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))
        
        if is_cross or (self.cur_step not in self.step_idx) or (self.cur_att_layer // 2 not in self.layer_idx):
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        

        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
    

        if len(self.cross_attns) == 0:
            out_c_target = self.attn_batch(qc[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[-num_heads:], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_target = self.attn_batch(qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[-num_heads:], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            mask = self.aggregate_cross_attn_map(idx_list=self.ref_token_idx) 
            mask_source = mask[-2] 
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask_ref = nn.functional.interpolate(mask_source.unsqueeze(0).unsqueeze(0), (res, res, len(self.ref_token_idx))).flatten()
            
            out_c_target = self.attn_batch(qc[:num_heads], kc[-num_heads:], vc[-num_heads:], sim[-num_heads:], attnc, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_target = self.attn_batch(qu[:num_heads], ku[-num_heads:], vu[-num_heads:], sim[-num_heads:], attnu, is_cross, place_in_unet, num_heads, **kwargs)

            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            
            out_c_target = out_c_target_fg
            out_u_target = out_u_target_fg
            
            self.self_attns_mask_ref = None
        
        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out

    def q_affine_mask(self,q,H,W):
        q_af=torch.zeros_like(q)
        for i in range(len(self.ref_token_idx)):
            save_image(q[:1,1,:,:], os.path.join(self.mask_save_dir, f"q_source_step_{self.cur_step}_layer_{self.cur_att_layer}_index_{i}.png"))
            q_i=q.masked_fill(~self.mask_ref[:,:,:,:,i].type(torch.bool),0)
            q_af=q_af+self.q_affine(i,q_i,H,W)
        return q_af


    def q_affine(self,i,q,H,W):     
        save_image(q[:1,1,:,:], os.path.join(self.mask_save_dir, f"q_mask_source_step_{self.cur_step}_layer_{self.cur_att_layer}_index_{i}.png"))
        translate=int(H*self.object_afflines[i]["translate"][0]), int(W*self.object_afflines[i]["translate"][1])
        q_af=functional.affine(q, self.object_afflines[i]["angle"], translate, self.object_afflines[i]["scale"], self.object_afflines[i]["shear"])
        return  q_af


    def creat_mask(self):
        for i in range(len(self.ref_token_idx)):
            mask_i=self.mask_ref[:,:,:,:,i]
            mask_i[mask_i>=self.thres[i]]=1
            mask_i[mask_i<self.thres[i]]=0
        return


    def save_masks(self,mask,mask_file_name):
        for i in range(len(self.ref_token_idx)):
            save_image(mask[:,:,:,:,i], os.path.join(self.mask_save_dir, mask_file_name+f"_step_{self.cur_step}_layer_{self.cur_att_layer}_index_{i}.png"))    
        return