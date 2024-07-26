import os
from typing import List
import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModel
from .attention_processor_modify_2 import AttnProcessor, IPAttnProcessor, AttnProcessorCross
import numpy as np

class modified_ImageProjModel(torch.nn.Module):
    """modified Projection Model, this one is more reasonable"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, number_fine_features=7):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.number_fine_features = number_fine_features
        self.number_img_tokens=257
        self.clip_embeddings_dim=clip_embeddings_dim

        self.proj1 = torch.nn.ModuleList([])
        for _ in range(number_fine_features):
            self.proj1.append(
                torch.nn.Linear(clip_embeddings_dim, cross_attention_dim,bias=False)
            )

        self.proj2 = torch.nn.Linear(self.number_fine_features, 1,bias=False)
        # after proj, the size will be 1*(4*768) or 1*(16*768)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        self.image=None
        self.controlnet_conditioning_scale=None

    def forward(self, image_embeds):
        rescale_embeds=[]
        for i in range(self.number_fine_features):
            rescale_embeds.append(self.proj1[i](image_embeds[:,i,:,:]))# (num_feature,bsz,257,768)
        rescale_embeds=torch.stack(rescale_embeds).permute(1,2,3,0)
        mean_embeds=self.proj2(rescale_embeds).squeeze(dim=-1)
        clip_extra_context_tokens = self.norm(mean_embeds)
        return clip_extra_context_tokens

    def visual(self,image_embeds):
        rescale_embeds = []
        for i in range(self.number_fine_features):
            rescale_embeds.append(self.proj1[i](image_embeds[:, i, :, :]))  # (num_feature,bsz,257,768)
        rescale_embeds = torch.stack(rescale_embeds).permute(1, 2, 3, 0)
        return rescale_embeds

class ImageProjModel(torch.nn.Module):
    """IP-adapter img-proj model input bsz*1024->bsz* 4OR16 *768 (FOR CLIP-V/L-14)"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        # after proj, the size will be 1*(4*768)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class IPAdapter_Ctrlnet_inference:
    def __init__(self, sd_pipe, ip_ctrlnet_ckpt, device, image_encoder_path=None, num_tokens=None,
                 number_fine_features=None, new_img_proj=None):
        self.model_type = sd_pipe.dtype
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ctrlnet_ckpt = ip_ctrlnet_ckpt
        self.num_tokens = num_tokens
        self.number_fine_features = number_fine_features
        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter(processor_to_keep_blocks_id=[13, 15, 17, 19, 21, 23, 25, 27, 29])


        assert new_img_proj in [0, 1, 2, 3, 4,
                                5,6]  # 0: original 1: SSR 2: resampler 3.modified SSR 4.SSR_pcver 5.detailed modified SSR 6 ssr with resampler
        self.new_img_proj = new_img_proj

        if image_encoder_path is not None:
            self.image_encoder = CLIPVisionModel.from_pretrained(self.image_encoder_path).to(self.device,
                                                                                             dtype=self.model_type)
        self.clip_image_processor = CLIPImageProcessor()


        if self.new_img_proj == 0:
            self.image_proj_model = self.init_proj()

        elif self.new_img_proj == 3 or self.new_img_proj == 5:  # modified ssr
            self.image_proj_model = self.init_proj_modified()

        self.image=None,
        self.control_image=None,
        self.mask_image=None,
        self.strength=None,
        self.controlnet_conditioning_scale=None,

        self.load_ip_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=1024 * (1 + self.number_fine_features),
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=self.model_type)
        return image_proj_model

    def init_proj_modified(self):
        image_proj_model = modified_ImageProjModel(cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                                                   clip_embeddings_dim=1024,
                                                   number_fine_features=self.number_fine_features).to(self.device,
                                                                                                      dtype=self.model_type)
        return image_proj_model
    def set_ip_adapter(self,processor_to_keep_blocks_id):
        unet = self.pipe.unet
        attn_procs = {}
        for b_i,name in enumerate(unet.attn_processors.keys()):
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            elif b_i in processor_to_keep_blocks_id:
                if "motion_module" in name:
                    attn_procs[name] = AttnProcessor()
                    continue
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.model_type)
            else:
                if "motion_module" in name:
                    attn_procs[name] = AttnProcessor()
                    continue
                attn_procs[name] = AttnProcessorCross(
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.model_type)

        unet.set_attn_processor(attn_procs)


    def load_ip_adapter(self):

        if os.path.splitext(self.ip_ctrlnet_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj_model": {}, "ip_weights": {}, "controlnet": {}}
            state_dict_for_adapter = {}
            state_dict_for_feature_resampler={}
            with safe_open(self.ip_ctrlnet_ckpt, framework="pt", device="cpu") as f:
            # with open(self.ip_ctrlnet_ckpt, framework="pt", device="cpu") as f:
                # k_nums=0
                # v_nums=0
                for key in f.keys():
                    # print(key)
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("controlnet"):
                        state_dict["controlnet"][key.replace("controlnet.", "")] = f.get_tensor(key)
                    elif key.split('.')[-2] == "to_k_ip" or key.split('.')[-2] == "to_v_ip":
                        # print(key,f.get_tensor(key).shape)
                        k_or_v = key.split('.')[-2][-4]
                        if key.startswith("unet.down_blocks."):
                            n1, n2 = int(key.split('.')[2]), int(key.split('.')[4])
                            state_dict_for_adapter[f"{2 * (n1 * 2 + n2) + 1}.to_{k_or_v}_ip.weight"] = f.get_tensor(key)
                            # print(f"{2*(n1*2+n2)+1}.to_{k_or_v}_ip.weight")
                        elif key.startswith("unet.up_blocks."):
                            n1, n2 = int(key.split('.')[2]), int(key.split('.')[4])
                            state_dict_for_adapter[
                                f"{12 + 2 * ((n1 - 1) * 3 + n2) + 1}.to_{k_or_v}_ip.weight"] = f.get_tensor(key)
                            # print(f"{12+2 * ((n1-1) * 3 + n2) + 1}.to_{k_or_v}_ip.weight")
                        elif key.startswith("unet.mid_block."):
                            state_dict_for_adapter[f"31.to_{k_or_v}_ip.weight"] = f.get_tensor(key)
                            # print(f"31.to_{k_or_v}_ip.weight")
                    elif key.startswith("feature_resampler"):
                        state_dict_for_feature_resampler[key.replace("feature_resampler.","")]=f.get_tensor(key)
        else:
            exit("no no no")
            state_dict = torch.load(ckpt_path, map_location="cpu")
        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        # for ip_layer,param in ip_layers.named_parameters():
        #     print(ip_layer,param.shape)
        ip_layers.load_state_dict(state_dict_for_adapter, strict=False)



        print(f"Successfully loaded weights from checkpoint {self.ip_ctrlnet_ckpt}")


    @torch.inference_mode()
    def get_image_embeds_new(self, pil_image=None,sort_attn=None):
        # todo get fine-grained feature(not mean)
        multi=False
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            elif isinstance(pil_image,List):
                multi = len(pil_image)
                pass
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values.to(
                self.device, dtype=self.model_type)
            # todo use hook to extract intermediate output of model
            inps, fine_grain_outs_ = [], []

            def layer_hook(module, inp, out):
                # inps.append(inp[0].data)
                fine_grain_outs_.append(out.data)

            # todo extract, follow ssr-encoder, use the last 6 layers
            hooks = []
            for layer_no in range(24 - self.number_fine_features, 24):
                hooks.append(
                    self.image_encoder.vision_model.encoder.layers[layer_no].layer_norm2.register_forward_hook(
                        layer_hook))
            pooler_outputs = self.image_encoder(clip_image).pooler_output
            for hook in hooks:
                hook.remove()
            image_embeds = torch.stack(fine_grain_outs_, dim=1)

            if multi==False:
                image_prompt_embeds = self.image_proj_model(image_embeds)
                uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
            else:
                image_prompt_embeds = self.image_proj_model(image_embeds)
                uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
                # a=image_prompt_embeds.permute(2,0,1).reshape(-1,len(pil_image)*self.num_tokens).permute(1,0).unsqueeze(dim=0)
                # b=torch.cat((image_prompt_embeds[0,:,:],image_prompt_embeds[1,:,:]),dim=0).unsqueeze(dim=0)
                image_prompt_embeds=torch.cat([image_prompt_embeds[i,:,:] for i in range(len(pil_image))],dim=0).unsqueeze(dim=0)
                uncond_image_prompt_embeds =torch.cat([uncond_image_prompt_embeds[i,:,:] for i in range(len(pil_image))],dim=0).unsqueeze(dim=0)
                self.set_multi(multi)
                pass

            return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_new_detail(self, pil_image=None, det_sca_r=2, det_sca_c=2):
        if pil_image is not None:
            multi = False
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            elif isinstance(pil_image, List):
                multi = len(pil_image)

            # todo clip image
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values.to(
                self.device, dtype=self.model_type) #(multi,3,224,224)

            # todo clip detail image prompt
            bsz=len(pil_image)
            clip_detail_image = [p_img.resize((224 * det_sca_r, 224 * det_sca_c)) for p_img in pil_image]
            clip_detail_image = [np.asarray(c_detail_img) for c_detail_img in clip_detail_image] #[(448,448,3 or 4),(448,448,3 or 4),.....]


            c_img_list = []
            for b_i in range(bsz):
                for r in range(det_sca_r):
                    for c in range(det_sca_c):
                        c_img_list.append(Image.fromarray(clip_detail_image[b_i][r * 224:(r + 1) * 224, c * 224:(c + 1) * 224]))
            clip_detail_image=self.clip_image_processor(images=c_img_list, return_tensors="pt").pixel_values.to(
                    self.device, dtype=self.model_type) #(multi*4,3,224,224)

            clip_image=torch.chunk(clip_image,multi,dim=0)
            clip_detail_image=torch.chunk(clip_detail_image,multi,dim=0)
            clip_detail_image_list = [torch.cat((c_img, c_detail_img), dim=0) for c_img,c_detail_img in zip(clip_image,clip_detail_image)] #[(5*224*224*3),(5*224*224*3),...]
            clip_detail_image=torch.cat(clip_detail_image_list,dim=0) #(multi*5,3,224,224)

            # todo use hook to extract intermediate output of model
            inps, fine_grain_outs_ = [], []

            def layer_hook(module, inp, out):
                # inps.append(inp[0].data)
                fine_grain_outs_.append(out.data)

            # todo extract, follow ssr-encoder, use the last 6 layers
            hooks = []
            for layer_no in range(24 - self.number_fine_features, 24):
                hooks.append(
                    self.image_encoder.vision_model.encoder.layers[layer_no].layer_norm2.register_forward_hook(
                        layer_hook))

            pooler_outputs = self.image_encoder(clip_detail_image.reshape(-1, 3, 224, 224)).pooler_output
            for hook in hooks:
                hook.remove()
            image_embeds_oris = torch.stack(fine_grain_outs_, dim=1)  # (bsz*5)*7*257*1024

            # todo for detail clip image
            image_embeds_oris=torch.chunk(image_embeds_oris,multi,dim=0)
            image_embeds=[]
            for image_embeds_ori in image_embeds_oris:
                image_embeds_ori = image_embeds_ori.reshape(-1, 1 + det_sca_r * det_sca_c, self.number_fine_features,
                                                            257,
                                                            1024)  # (multi*5,7,257,1024)
                image_embeds_ori_glob = image_embeds_ori[:, 0, :, 0:1, :]  # (bsz,7,1,1024)
                image_embeds_ori_detail = image_embeds_ori[:, 1:, :, 1:, :].reshape(-1, det_sca_r * det_sca_c,
                                                                                    self.number_fine_features, 16, 16,
                                                                                    1024)  # (bsz,4,7,16,16,1024)
                image_embeds_ori_det_rs = []
                for r in range(det_sca_r):
                    image_embeds_ori_det_r = []
                    for c in range(det_sca_c):
                        image_embeds_ori_det_r.append(image_embeds_ori_detail[:, r * det_sca_c + c])
                    image_embeds_ori_det_r = torch.cat(image_embeds_ori_det_r, dim=3)
                    image_embeds_ori_det_rs.append(image_embeds_ori_det_r)
                image_embeds_ori_detail = torch.cat(image_embeds_ori_det_rs, dim=2)

                image_embeds_ori_det = image_embeds_ori_detail.reshape(-1, self.number_fine_features,
                                                                       16 * det_sca_c * 16 * det_sca_r,
                                                                       1024)
                image_embeds.append(torch.cat((image_embeds_ori_glob, image_embeds_ori_det), dim=2))  # bss*7*1025*1024
                pass
            image_embeds=torch.cat(image_embeds,dim=0)
            if multi==False:
                image_prompt_embeds = self.image_proj_model(image_embeds)
                uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
            else:
                image_prompt_embeds = self.image_proj_model(image_embeds)
                uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
                image_prompt_embeds=torch.cat([image_prompt_embeds[i,:,:] for i in range(multi)],dim=0).unsqueeze(dim=0)
                uncond_image_prompt_embeds =torch.cat([uncond_image_prompt_embeds[i,:,:] for i in range(multi)],dim=0).unsqueeze(dim=0)
                self.set_multi(multi)
            return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def set_multi(self,multi):
        for processor in self.pipe.unet.attn_processors.values():
            if isinstance(processor, IPAttnProcessor) or isinstance(processor,AttnProcessorCross):
                processor.multi = multi

    def set_word_attn_mask(self, word_attn_mask):
        if word_attn_mask is None:
            for attn_processor in self.pipe.unet.attn_processors.values():
                if isinstance(attn_processor, IPAttnProcessor):
                    attn_processor.word_attn_mask = None
        else:
            scales = [ 4, 4, 4, 2, 2, 2, 1, 1, 1]
            p_id = 0
            tn, h, w = word_attn_mask.shape
            for attn_processor in self.pipe.unet.attn_processors.values():
                if isinstance(attn_processor, IPAttnProcessor):
                    attn_processor.word_attn_mask = torch.max_pool2d(word_attn_mask, kernel_size=scales[p_id],
                                                                     stride=scales[p_id]).reshape(-1, h * w // (
                            scales[p_id] ** 2)).permute(1, 0)

                    # attn_processor.word_attn_mask =torch.nn.AvgPool2d(scales[p_id], stride=scales[p_id])(word_attn_mask).reshape(-1,h * w // (scales[p_id] ** 2)).permute(1, 0)
                    p_id += 1

    def set_img_prompt_attn_mask(self, img_prompt_attn_mask):
        if img_prompt_attn_mask is None:
            for attn_processor in self.pipe.unet.attn_processors.values():
                if isinstance(attn_processor, IPAttnProcessor):
                    attn_processor.img_prompt_attn_mask = None
        else:
            scales = [4, 4, 4, 2, 2, 2, 1, 1, 1]
            p_id = 0
            if isinstance(img_prompt_attn_mask, List):
                tn, h, w = img_prompt_attn_mask[0].shape
                for attn_processor in self.pipe.unet.attn_processors.values():
                    if isinstance(attn_processor, IPAttnProcessor):
                        # attn_processor.img_prompt_attn_mask = [torch.max_pool2d(i_p_a_m,
                        #                                                         kernel_size=scales[p_id],
                        #                                                         stride=scales[p_id]).reshape(-1,h * w // (scales[p_id] ** 2)).permute(1, 0) for i_p_a_m in img_prompt_attn_mask]
                        attn_processor.img_prompt_attn_mask = [torch.nn.AvgPool2d(scales[p_id],stride=scales[p_id])(i_p_a_m).reshape(-1,
                                                                                                             h * w // (
                                                                                                                         scales[
                                                                                                                             p_id] ** 2)).permute(
                            1, 0) for i_p_a_m in img_prompt_attn_mask]

                        p_id += 1
            else:
                tn, h, w = img_prompt_attn_mask.shape
                for attn_processor in self.pipe.unet.attn_processors.values():
                    if isinstance(attn_processor, IPAttnProcessor):
                        attn_processor.img_prompt_attn_mask = torch.max_pool2d(img_prompt_attn_mask,
                                                                               kernel_size=scales[p_id],
                                                                               stride=scales[p_id]).reshape(-1,h * w // (scales[p_id] ** 2)).permute(1, 0)

                        # attn_processor.img_prompt_attn_mask = torch.nn.AvgPool2d(scales[p_id], stride=scales[p_id])(img_prompt_attn_mask).reshape(-1,h * w // (scales[p_id] ** 2)).permute(1, 0)

                        p_id += 1

    def generate(
            self,
            pil_image=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=None,
            width=None,
            img_prompt_attn_mask=None,
            num_frames=16,
            **kwargs,
    ):
        self.set_scale(scale)


        num_prompts = 1

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        if self.new_img_proj in [1, 3, 4]:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_new(
                pil_image=pil_image,
            )
        elif self.new_img_proj == 5:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_new_detail(
                pil_image=pil_image
            )

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        bs_embed, u_seq_len, _ = uncond_image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, u_seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1).to(self.device,
                                                                                       dtype=self.model_type)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1).to(
                self.device, dtype=self.model_type)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        # set ip mask
        if isinstance(img_prompt_attn_mask, List):
            i_p_a_ms = [i_p_a_m.permute(1, 0).reshape(-1, height // 8, width // 8) for i_p_a_m in
                        img_prompt_attn_mask]
            self.set_img_prompt_attn_mask(i_p_a_ms)
        else:
            self.set_img_prompt_attn_mask(
                img_prompt_attn_mask.permute(1, 0).reshape(-1, height // 8, width // 8))
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,  # (2,77+num_tokens,768)
            height=height,
            width=width,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=self.image,
            control_image=self.control_image,
            mask_image=self.mask_image,
            strength=self.strength,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            num_frames=num_frames,
            **kwargs,
        ).images

        return images

