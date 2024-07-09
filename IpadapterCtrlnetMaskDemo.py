import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
import argparse
import numpy as np
import torch
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, UNet2DConditionModel, StableDiffusionControlNetPipeline, TCDScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from ip_adapter.ip_adapter_ctrlnet_inference_4 import IPAdapter_Ctrlnet_inference
from diffusers.utils import load_image
import gradio as gr
from gradio.components.image_editor import Brush
import cv2
from huggingface_hub import hf_hub_download
import base64
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="stablediffusionapi/anything-v5",required=False)
    parser.add_argument("--controlnet_path",type=str,default="lllyasviel/sd-controlnet-openpose",required=False)
    parser.add_argument("--image_encoder_path",type=str,default="openai/clip-vit-large-patch14",required=False)
    parser.add_argument("--tokennumbers",type=int,default=1025,required=False)
    parser.add_argument("--num_inference_steps",type=str,default=12,required=False)
    parser.add_argument("--img_p_scale",type=float,default=0.7,required=False)
    parser.add_argument("--ctrl_scale",type=float,default=0.8,required=False)
    parser.add_argument("--txt_p_scale",type=float,default=5,required=False)
    parser.add_argument("--seed",type=int,default=42,required=False)
    parser.add_argument("--height",type=int,default=512,required=False)
    parser.add_argument("--width",type=int,default=512,required=False)
    parser.add_argument(
        "--adapter_path_if_ignore",
        default=["models/final_60000.safetensors",True]
    )
    parser.add_argument("--img_prompt_mask",default={"1_body": []})
    parser.add_argument("--img_prompt_mask_scale",default={"0_global":1,"1_body":1})# first value for whole, second value for mask
    args = parser.parse_args()
    return args

pipe = None
ip_model = None

def preload():
    global pipe, ip_model
    torch.cuda.empty_cache()
    
    # noise_scheduler = DDIMScheduler(
    #     num_train_timesteps=1000,
    #     beta_start=0.00085,
    #     beta_end=0.012,
    #     beta_schedule="scaled_linear",
    #     clip_sample=False,
    #     set_alpha_to_one=False,
    #     steps_offset=1,
    # )
    noise_scheduler = TCDScheduler()
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    
    # load controlnet
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
        
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    ).to(dtype=torch.float16)
    
    pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-12steps-CFG-lora.safetensors"))
    
    ip_ctrlnet_ckpt = args.adapter_path_if_ignore[0]
    new_img_proj = 0
    if args.tokennumbers == 257:
        new_img_proj = 3
    elif args.tokennumbers == 1025:
        new_img_proj = 5

    # todo model
    ip_model = IPAdapter_Ctrlnet_inference(
        pipe, ip_ctrlnet_ckpt, 
        "cuda",
        image_encoder_path=args.image_encoder_path,
        num_tokens=args.tokennumbers,
        number_fine_features=7,
        new_img_proj=new_img_proj)
    
def prepare_ref_refmask(imginput, latentmask):
    # todo ip mask
    maskinput = imginput["layers"][0]
    maskgray = maskinput[:, :, 3] / 255.0
    total_sum = np.sum(maskgray)

    if total_sum == 0:
        maskindices = np.array(range(args.tokennumbers))
    else:
        args.img_prompt_mask_scale["0_global"] = 0
        # Add padding
        top, bottom, left, right = 0, 0, 0, 0
        border_type = cv2.BORDER_CONSTANT
        value = [0]  # Padding color for constant border type (black)
        m_width, m_height = maskgray.shape
        if m_width > m_height:
            diff = round((m_width - m_height) / 2)
            top, bottom = diff, diff
            maskgray = cv2.copyMakeBorder(maskgray, top, bottom, left, right, border_type, value=value)
        elif m_height > m_width:
            diff = round((m_height - m_width) / 2)
            left, right = diff, diff
            maskgray = cv2.copyMakeBorder(maskgray, top, bottom, left, right, border_type, value=value)
        if args.tokennumbers == 257:
            mask16x16 = cv2.resize(maskgray, [16, 16])
            maskflat = mask16x16.flatten()
            maskindices = np.nonzero(maskflat)[0] + 1  # because index 0 reserved for global feature
        elif args.tokennumbers == 1025:
            mask32x32 = cv2.resize(maskgray, [32, 32])
            maskflat = mask32x32.flatten()
            maskindices = np.nonzero(maskflat)[0] + 1  # because index 0 reserved for global feature
    args.img_prompt_mask["1_body"] = maskindices.tolist()

    # todo set image prompt strength
    strengthen_id_scales_2 = np.tile(np.array([args.img_prompt_mask_scale["0_global"]], dtype=np.float16), (args.tokennumbers,))
    
    # todo latent mask
    maskgray_latent = latentmask["layers"][0][:, :, 3] / 255.0
    total_sum_latent = np.sum(maskgray_latent)
    if total_sum_latent == 0:
        latentmask_weight = torch.ones(((args.height // 8) * (args.width // 8),), dtype=torch.float16).to(
            device=ip_model.device)
    else:
        maskgray_latent = cv2.resize(maskgray_latent, [(args.height // 8), (args.width // 8)])
        maskgray_latent = maskgray_latent.flatten()
        maskindices_latent = np.nonzero(maskgray_latent)
        latentmask_weight = torch.zeros(((args.height // 8) * (args.width // 8),), dtype=torch.float16).to(
            device=ip_model.device)
        latentmask_weight[maskindices_latent] = 1

    for query, m_id in args.img_prompt_mask_scale.items():
        if query == "0_global": continue
        strengthen_id_scales_2[np.array(args.img_prompt_mask[query])] = m_id
    strengthen_id_scales_2 = strengthen_id_scales_2.tolist()

    # todo ip attn mask
    img_prompt_attn_mask = torch.tensor(strengthen_id_scales_2, dtype=torch.float16).repeat(
        (args.height // 8) * (args.width // 8), 1).to(device=ip_model.device)  # (64*64)*257

    # todo then apply latent mask
    img_prompt_attn_mask = img_prompt_attn_mask.permute(1, 0) * latentmask_weight
    img_prompt_attn_mask = img_prompt_attn_mask.permute(1, 0)

    image = imginput["background"][:, :, 0:3]
    r_width, r_height, _ = image.shape

    # if image is not a square image, square it
    # Add padding
    top, bottom, left, right = 0, 0, 0, 0
    border_type = cv2.BORDER_CONSTANT
    value = [0, 0, 0]  # Padding color for constant border type (black)
    if r_width != r_height and r_width > r_height:
        diff = round((r_width - r_height) / 2)
        top, bottom = diff, diff
        image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=value)
    elif r_width != r_height and r_height > r_width:
        diff = round((r_height - r_width) / 2)
        left, right = diff, diff
        image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=value)
    
    return Image.fromarray(image), img_prompt_attn_mask

def test(
        stepnum, 
        seednum, 
        pos, 
        neg,
        width,
        height,
        img_p_scale, 
        ctrl_scale, 
        txt_p_scale, 
        imginput,
        latentmask,
        apply_ref2,
        imginput2,
        latentmask2,
        apply_ref3,
        imginput3,
        latentmask3,
    ):
    
    global pipe, ip_model
    if pipe == None:
        preload()
    
    args.num_inference_steps = stepnum
    args.seed = seednum
    args.img_p_scale = float(img_p_scale)
    args.ctrl_scale = float(ctrl_scale)
    args.txt_p_scale = float(txt_p_scale)
    args.height = height
    args.width = width
    
    condition_image = Image.fromarray(latentmask["background"][:, :, 0:3])
    processed_control_condition = condition_image.resize((args.height, args.width))

    pil_images=[]
    img_prompt_attn_masks=[]

    pil_image, img_prompt_attn_mask=prepare_ref_refmask(imginput, latentmask)
    pil_images.append(pil_image)
    img_prompt_attn_masks.append(img_prompt_attn_mask)
    
    if apply_ref2:
        pil_image2, img_prompt_attn_mask2=prepare_ref_refmask(imginput2, latentmask2)
        pil_images.append(pil_image2)
        img_prompt_attn_masks.append(img_prompt_attn_mask2)
        
    if apply_ref3:
        pil_image3, img_prompt_attn_mask3=prepare_ref_refmask(imginput3, latentmask3)
        pil_images.append(pil_image3)
        img_prompt_attn_masks.append(img_prompt_attn_mask3)

    if len(img_prompt_attn_masks)==1:
        img_prompt_attn_masks=img_prompt_attn_masks[0]

    ip_model.image=processed_control_condition
    ip_model.controlnet_conditioning_scale=args.ctrl_scale

    images = ip_model.generate(
        pil_image=pil_images,
        prompt=pos,
        negative_prompt=neg,
        img_prompt_attn_mask=img_prompt_attn_masks,
        to_attnprocessor=True,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        scale=args.img_p_scale,
        guidance_scale=args.txt_p_scale,
        height=args.height, 
        width=args.width)

    return images[0]

def base64_to_image(base64_str):
    if "base64" in base64_str:
        img_data = base64.b64decode(base64_str.split("base64,")[1])
        img = Image.open(BytesIO(img_data))
        return np.array(img)
    else:
        return None

def prepare_ref_refmask_base64(imginput_base64, latentmask_base64):
    global args
    imginput = base64_to_image(imginput_base64)
    latentmask = base64_to_image(latentmask_base64)
        
    if imginput is not None:
        if imginput.shape[2] == 4:
            maskgray = imginput[:, :, 3] / 255.0
        else:
            maskgray = np.ones((imginput.shape[0], imginput.shape[1]))  # No alpha channel, assume fully opaque
    else:
        maskgray = np.zeros((args.height // 8, args.width // 8))
    
    total_sum = np.sum(maskgray)

    if total_sum == 0:
        if args.tokennumbers == 257:
            maskindices = np.array(range(257))
        elif args.tokennumbers == 1025:
            maskindices = np.array(range(1025))
    else:
        args.img_prompt_mask_scale["0_global"] = 0
        # Add padding
        top, bottom, left, right = 0, 0, 0, 0
        border_type = cv2.BORDER_CONSTANT
        value = [0]  # Padding color for constant border type (black)
        m_width, m_height = maskgray.shape
        if m_width > m_height:
            diff = round((m_width - m_height) / 2)
            top, bottom = diff, diff
            maskgray = cv2.copyMakeBorder(maskgray, top, bottom, left, right, border_type, value=value)
        elif m_height > m_width:
            diff = round((m_height - m_width) / 2)
            left, right = diff, diff
            maskgray = cv2.copyMakeBorder(maskgray, top, bottom, left, right, border_type, value=value)
        if args.tokennumbers == 257:
            mask16x16 = cv2.resize(maskgray, (16, 16))
            maskflat = mask16x16.flatten()
            maskindices = np.nonzero(maskflat)[0] + 1  # because index 0 reserved for global feature
        elif args.tokennumbers == 1025:
            mask32x32 = cv2.resize(maskgray, (32, 32))
            maskflat = mask32x32.flatten()
            maskindices = np.nonzero(maskflat)[0] + 1  # because index 0 reserved for global feature
    args.img_prompt_mask["1_body"] = maskindices.tolist()

    # Set image prompt strength
    strengthen_id_scales_2 = np.tile(np.array([args.img_prompt_mask_scale["0_global"]], dtype=np.float16), (args.tokennumbers,))

    # Latent mask
    if latentmask is not None:
        if latentmask.shape[2] == 4:
            maskgray_latent = latentmask[:, :, 3] / 255.0
        else:
            maskgray_latent = np.ones((latentmask.shape[0], latentmask.shape[1]))  # No alpha channel, assume fully opaque
    else:
        maskgray_latent = np.zeros((args.height // 8, args.width // 8))
    total_sum_latent = np.sum(maskgray_latent)
    if total_sum_latent == 0:
        latentmask_weight = torch.ones(((args.height // 8) * (args.width // 8),), dtype=torch.float16).to(device=ip_model.device)
    else:
        maskgray_latent = cv2.resize(maskgray_latent, (args.height // 8, args.width // 8))
        maskgray_latent = maskgray_latent.flatten()
        maskindices_latent = np.nonzero(maskgray_latent)
        latentmask_weight = torch.zeros(((args.height // 8) * (args.width // 8),), dtype=torch.float16).to(device=ip_model.device)
        latentmask_weight[maskindices_latent] = 1

    for query, m_id in args.img_prompt_mask_scale.items():
        if query == "0_global":
            continue
        strengthen_id_scales_2[np.array(args.img_prompt_mask[query])] = m_id
    strengthen_id_scales_2 = strengthen_id_scales_2.tolist()

    # Image prompt attention mask
    img_prompt_attn_mask = torch.tensor(strengthen_id_scales_2, dtype=torch.float16).repeat(
        (args.height // 8) * (args.width // 8), 1).to(device=ip_model.device)

    # Apply latent mask
    img_prompt_attn_mask = img_prompt_attn_mask.permute(1, 0) * latentmask_weight
    img_prompt_attn_mask = img_prompt_attn_mask.permute(1, 0)

    return img_prompt_attn_mask

def generate(
        stepnum, 
        seednum, 
        pos, 
        neg,
        width,
        height,
        img_p_scale, 
        ctrl_scale, 
        txt_p_scale, 
        
        condition_base64,
        
        ref_base64,
        ref_mask_base64,
        condition_mask_base64,
        
        ref2_base64,
        ref2_mask_base64,
        condition2_mask_base64
    ):
    
    global pipe, ip_model
    if pipe == None:
        preload()
        
    args.num_inference_steps = stepnum
    args.seed = seednum
    args.img_p_scale = float(img_p_scale)
    args.ctrl_scale = float(ctrl_scale)
    args.txt_p_scale = float(txt_p_scale)
    args.height = height
    args.width = width
    
    condition_image = Image.fromarray(base64_to_image(condition_base64))
    processed_control_condition = condition_image.resize((height, width))

    pil_images=[]
    img_prompt_attn_masks=[]
    
    pil_images.append(Image.fromarray(base64_to_image(ref_base64)))
    
    if ref2_base64 and ref2_base64 != "":
        pil_images.append(Image.fromarray(base64_to_image(ref2_base64)))

    img_prompt_attn_masks.append(prepare_ref_refmask_base64(ref_mask_base64, condition_mask_base64))
    
    if condition2_mask_base64 and condition2_mask_base64 != "":
        img_prompt_attn_masks.append(prepare_ref_refmask_base64(ref2_mask_base64, condition2_mask_base64))
    
    if len(img_prompt_attn_masks)==1:
        img_prompt_attn_masks=img_prompt_attn_masks[0]


    ip_model.image=processed_control_condition
    ip_model.controlnet_conditioning_scale=args.ctrl_scale

    images = ip_model.generate(
        pil_image=pil_images,
        prompt=pos,
        negative_prompt=neg,
        img_prompt_attn_mask=img_prompt_attn_masks,
        to_attnprocessor=True,
        num_samples=1,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        scale=args.img_p_scale,
        guidance_scale=args.txt_p_scale,
        height=args.height, 
        width=args.width)

    return images[0]

def toggle_visibility(apply_ref):
    return gr.update(visible=apply_ref)

def update_latent_masks(img):
    return img, img, img

def main():
    global args
    args = parse_args()
    with gr.Blocks() as demo:
        demo.title = "Ami-adapter"
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Settings", open=False):
                    c_pos = gr.Textbox(label="Positive prompts", value="a photo of a girl on the beach")
                    c_neg = gr.Textbox(label="Negative prompts", value="ugly, monochrome, two people")
                    with gr.Row():
                        c_stepnum = gr.Number(label="Steps", value=args.num_inference_steps, minimum=1, maximum=100)
                        c_seednum = gr.Number(label="Seed", value=42, minimum=0)
                        c_width = gr.Number(label="Width", value=512, minimum=8, maximum=2048)
                        c_height = gr.Number(label="Height", value=512, minimum=8, maximum=2048)
                    with gr.Row():
                        img_p_scale = gr.Slider(label="img_p_scale", value=args.img_p_scale, minimum=0.0, maximum=1.0)
                        ctrl_scale = gr.Slider(label="ctrl_scale", value=args.ctrl_scale, minimum=0.0, maximum=1.0)
                        txt_p_scale = gr.Slider(label="txt_p_scale", value=args.txt_p_scale, minimum=0.0, maximum=10.0)
                    condition_img = gr.Image(label="Condition Image", type="pil")
                
                with gr.Group():
                    with gr.Row():
                        imginput = gr.ImageMask(label="Reference img", brush=Brush(colors=["#FFFFFF"]))
                        latentmask = gr.ImageMask(label="Latent mask", brush=Brush(colors=["#FFFFFF"]))
                        
                with gr.Group():
                    apply_ref2 = gr.Checkbox(label="Apply 2nd Ref", value=False) 
                    with gr.Row(visible=False) as ref2_block:
                        imginput2 = gr.ImageMask(label="Reference img", brush=Brush(colors=["#FFFFFF"]))
                        latentmask2 = gr.ImageMask(label="Latent mask", brush=Brush(colors=["#FFFFFF"]))
                apply_ref2.change(fn=toggle_visibility, inputs=apply_ref2, outputs=ref2_block)
                
                with gr.Group():
                    apply_ref3 = gr.Checkbox(label="Apply 3rd Ref", value=False)
                    with gr.Row(visible=False) as ref3_block:
                        imginput3 = gr.ImageMask(label="Reference img", brush=Brush(colors=["#FFFFFF"]))
                        latentmask3 = gr.ImageMask(label="Latent mask", brush=Brush(colors=["#FFFFFF"]))
                    apply_ref3.change(fn=toggle_visibility, inputs=apply_ref3, outputs=ref3_block)
                
                condition_img.change(fn=update_latent_masks, inputs=condition_img, outputs=[latentmask, latentmask2, latentmask3])
            
            with gr.Column():
                out = gr.Image(label="Generated image", type="pil")
        
        runBtn = gr.Button(value="Run")
        
        inputs = [
            c_stepnum, 
            c_seednum, 
            c_pos, 
            c_neg,
            c_width,
            c_height,
            img_p_scale, 
            ctrl_scale, 
            txt_p_scale,
            imginput,
            latentmask,
            apply_ref2,
            imginput2,
            latentmask2,
            apply_ref3,
            imginput3,
            latentmask3,
        ]
        outputs = [out]
        
        runBtn.click(test, inputs=inputs, outputs=outputs)
        
        
        with gr.Accordion("API", open=False):
            condition_base64 = gr.Textbox(label="Condition Image base64")
            ref_base64 = gr.Textbox(label="Reference Image base64")
            ref_mask_base64 = gr.Textbox(label="Reference Image Mask base64")
            condition_mask_base64 = gr.Textbox(label="Condition Image Mask base64")
            ref2_base64 = gr.Textbox(label="Reference Image base64")
            ref2_mask_base64 = gr.Textbox(label="Reference Image Mask base64")
            condition2_mask_base64 = gr.Textbox(label="Condition Image Mask base64")
            
            generate_inputs = [
                c_stepnum, 
                c_seednum, 
                c_pos, 
                c_neg,
                c_width,
                c_height,
                img_p_scale, 
                ctrl_scale, 
                txt_p_scale,
                
                condition_base64,
                
                ref_base64,
                ref_mask_base64,
                condition_mask_base64,
                
                ref2_base64,
                ref2_mask_base64,
                condition2_mask_base64,
            ]
            
            apiBtn = gr.Button(value="API")
            apiBtn.click(generate, inputs=generate_inputs, outputs=outputs, api_name="generate")
    
    demo.queue(max_size=1)
    demo.launch(share=True, server_port=7860)

if __name__ == '__main__':
    main()
    