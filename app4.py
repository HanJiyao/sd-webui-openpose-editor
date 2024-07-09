import os
import argparse
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
from diffusers import (DDIMScheduler,AutoencoderKL, ControlNetModel,UNet2DConditionModel,StableDiffusionInpaintPipelineLegacy,
                       StableDiffusionControlNetPipeline,StableDiffusionControlNetInpaintPipeline,StableDiffusionPipeline)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import datetime
from ip_adapter.ip_adapter_ctrlnet_inference_4 import IPAdapter_Ctrlnet_inference
from diffusers.utils import load_image
import gradio as gr
from gradio.components.image_editor import Brush
import cv2
from controlnet_aux import OpenposeDetector
from controlnet_aux import MidasDetector


def condi_gen(ctrlnet,poseinput,conditionornot):
    args.control_condition = poseinput
    args.controlnet_path = ctrlnet
    # todo condition image
    if conditionornot == "Normal":
        condition_image = load_image(args.control_condition)
        if "openpose" in args.controlnet_path:
            openpose_generator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            processed_control_condition = openpose_generator(condition_image, include_face=False,
                                                                 include_hand=False)
            # processed_control_condition = processed_control_condition.resize((args.height, args.width))
        elif "depth" in args.controlnet_path:
            depth_generator = MidasDetector.from_pretrained("lllyasviel/Annotators")
            processed_control_condition = depth_generator(condition_image)
            # processed_control_condition = processed_control_condition.resize((args.height, args.width))
    elif conditionornot == "Condition":
        processed_control_condition = load_image(args.control_condition)
        # processed_control_condition = processed_control_condition.resize((args.height, args.width))
    else:
        processed_control_condition = Image.fromarray(np.zeros([args.height, args.width, 3], dtype=np.uint8))
    return processed_control_condition,processed_control_condition,processed_control_condition

def test(bmodelpath, fmodelpath,tokennumbers, stepnum, seednum, pos, neg,width,height,strength,
         ctrlnet, conditionornot, inpaintornot,scale1, scale2, scale3, imginput1,latentmask1,apply_ref1,
         imginput2,latentmask2,apply_ref2,imginput3,latentmask3,apply_ref3,poseinput,inpaintedimage):
    torch.cuda.empty_cache()
    print("loading-----")
    args.num_inference_steps = stepnum
    args.seed = seednum
    args.pretrained_model_name_or_path = bmodelpath
    args.adapter_path_if_ignore[0] = "/home/yixuan/projects/comictitan/workspace/models/"+fmodelpath
    args.img_p_scale = float(scale1)
    args.ctrl_scale = float(scale2)
    args.txt_p_scale = float(scale3)
    args.height = height
    args.width = width
    args.controlnet_path=ctrlnet
    if tokennumbers == 257:
        args.number_of_tokens_proj_model=[257,3]
    elif tokennumbers == 1025:
        args.number_of_tokens_proj_model=[1025,5]

    if conditionornot == "None":
        args.ctrl_scale = 0.0
    else:
        args.control_condition = poseinput

    ip_ctrlnet_ckpt = args.adapter_path_if_ignore[0]
    device = "cuda"
    noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
    )

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(
            dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    if "openpose" in args.controlnet_path:
        openpose_generator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    elif "depth" in args.controlnet_path:
        depth_generator = MidasDetector.from_pretrained("lllyasviel/Annotators")
    # load controlnet
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    if inpaintornot and conditionornot in ["None","Normal", "Condition"]:
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
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
    # elif inpaintornot and conditionornot in ["None"]:
    #     pipe=StableDiffusionInpaintPipelineLegacy.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         controlnet=controlnet,
    #         torch_dtype=torch.float16,
    #         scheduler=noise_scheduler,
    #         unet=unet,
    #         tokenizer=tokenizer,
    #         text_encoder=text_encoder,
    #         vae=vae,
    #         feature_extractor=None,
    #         safety_checker=None,
    #     ).to(dtype=torch.float16)
    elif not inpaintornot and conditionornot in ["Normal", "Condition"]:
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
    elif not inpaintornot and conditionornot in ["None"]:
        pipe = StableDiffusionPipeline.from_pretrained(
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

    # todo model
    ip_model = IPAdapter_Ctrlnet_inference(pipe, ip_ctrlnet_ckpt, device,
                                           image_encoder_path=args.image_encoder_path,
                                           num_tokens=args.number_of_tokens_proj_model[0],
                                           number_fine_features=7,
                                           new_img_proj=args.number_of_tokens_proj_model[1])

    # # todo word attn mask
    # word_attn_mask = torch.tensor(strengthen_id_scales, dtype=torch.float16).repeat(
    #     (args.height // 8) * (args.width // 8), 1).to(device=ip_model.device)

    # todo condition image
    if conditionornot == "Normal":
        # todo it's an original condition image, we need to turn it
        condition_image = load_image(args.control_condition)
        if "openpose" in args.controlnet_path:
            processed_control_condition = openpose_generator(condition_image, include_face=False, include_hand=False)
            processed_control_condition = processed_control_condition.resize((args.height, args.width))
        elif "depth" in args.controlnet_path:
            processed_control_condition = depth_generator(condition_image)
            processed_control_condition = processed_control_condition.resize((args.height, args.width))
    elif conditionornot == "Condition":
        condition_image = load_image(args.control_condition)
        processed_control_condition = condition_image.resize((args.height, args.width))
    else:
        processed_control_condition = Image.fromarray(np.zeros([args.height, args.width,3], dtype=np.uint8))

    def prepare_ref_refmask(imginput,latentmask):
        # todo ip mask
        maskinput = imginput["layers"][0]
        maskgray = maskinput[:, :, 3] / 255.0
        total_sum = np.sum(maskgray)

        if total_sum == 0:
            if tokennumbers == 257:
                maskindices = np.array(range(257))
            elif tokennumbers == 1025:
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
            if tokennumbers == 257:
                mask16x16 = cv2.resize(maskgray, [16, 16])
                maskflat = mask16x16.flatten()
                maskindices = np.nonzero(maskflat)[0] + 1  # because index 0 reserved for global feature
            elif tokennumbers == 1025:
                mask32x32 = cv2.resize(maskgray, [32, 32])
                maskflat = mask32x32.flatten()
                maskindices = np.nonzero(maskflat)[0] + 1  # because index 0 reserved for global feature
        args.img_prompt_mask["1_body"] = maskindices.tolist()

        # todo set image prompt strength
        strengthen_id_scales_2 = np.tile(np.array([args.img_prompt_mask_scale["0_global"]], dtype=np.float16),
                                         (args.number_of_tokens_proj_model[0],))
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

    images=[]
    img_prompt_attn_masks=[]

    if apply_ref1:
        image1, img_prompt_attn_mask1=prepare_ref_refmask(imginput1,latentmask1)
        images.append(image1)
        img_prompt_attn_masks.append(img_prompt_attn_mask1)
    if apply_ref2:
        image2, img_prompt_attn_mask2=prepare_ref_refmask(imginput2,latentmask2)
        images.append(image2)
        img_prompt_attn_masks.append(img_prompt_attn_mask2)
    if apply_ref3:
        image3, img_prompt_attn_mask3=prepare_ref_refmask(imginput3,latentmask3)
        images.append(image3)
        img_prompt_attn_masks.append(img_prompt_attn_mask3)

    if len(img_prompt_attn_masks)==1:
        img_prompt_attn_masks=img_prompt_attn_masks[0]

    if inpaintornot and conditionornot in ["None","Normal", "Condition"]:
        image=Image.fromarray(inpaintedimage["background"][:,:,:3])
        image=image.resize((args.height,args.width))
        ip_model.image=image.convert("RGB")
        mask_image=Image.fromarray(inpaintedimage["layers"][0][:, :, 3])
        mask_image=mask_image.resize((args.height,args.width))

        ip_model.mask_image=mask_image.convert("RGB")
        ip_model.control_image=processed_control_condition
        print(image)
        image.save("yessss.png")
        print(mask_image)
        print(processed_control_condition)
        ip_model.strength=strength
        ip_model.controlnet_conditioning_scale=args.ctrl_scale
    # elif inpaintornot and conditionornot in ["None"]:
    #     ip_model.image = inpaintedimage["layers"][0]
    #     ip_model.mask_image = inpaintedimage["layers"][0][:, :, 3] / 255.0
    #     ip_model.strength = strength
    elif not inpaintornot and conditionornot in  ["Normal", "Condition"]:
        ip_model.image=processed_control_condition
        ip_model.controlnet_conditioning_scale=args.ctrl_scale
    elif not inpaintornot and conditionornot in ["None"]:
        pass


    images = \
        ip_model.generate(pil_image=images,
                          prompt=pos,
                          negative_prompt=neg,
                          # word_attn_mask=word_attn_mask,
                          img_prompt_attn_mask=img_prompt_attn_masks,
                          to_attnprocessor=True,
                          num_samples=1,
                          num_inference_steps=args.num_inference_steps,
                          seed=args.seed,
                          scale=args.img_p_scale,
                          guidance_scale=args.txt_p_scale,
                          height=args.height, width=args.width)[0]

    # images.save(os.path.join(args.output_dir, f"out.png"))
    return images, imginput1["background"][:,:,0:3],imginput2["background"][:,:,0:3],imginput3["background"][:,:,0:3]


def parse_args():
    output_dir = "output/test_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    pre_dir = output_dir.split("T")[0]
    post_dir = output_dir.split("T")[1]
    output_dir = os.path.join(pre_dir, post_dir)

    parser = argparse.ArgumentParser(description="Simple example of a test script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="stablediffusionapi/anything-v5",required=False)
    # parser.add_argument("--pretrained_model_name_or_path", type=str, default="emilianJR/AnyLORA")
    # parser.add_argument("--pretrained_model_name_or_path", type=str, default="hogiahien/anything-v5-edited")
    # parser.add_argument("--pretrained_model_name_or_path", type=str, default="Lykon/DreamShaper")
    # parser.add_argument("--pretrained_model_name_or_path", type=str, default="stablediffusionapi/counterfeit-v30")
    parser.add_argument("--controlnet_path",type=str,default="lllyasviel/sd-controlnet-openpose",required=False)
    # parser.add_argument("--controlnet_path",type=str,default="lllyasviel/control_v11f1p_sd15_depth",required=False)
    parser.add_argument("--output_dir",type=str,default=output_dir,required=False)
    parser.add_argument("--image_encoder_path",type=str,default="openai/clip-vit-large-patch14",required=False)
    parser.add_argument("--number_of_tokens_proj_model",type=list,default=[257,3],required=False,)
    parser.add_argument("--num_inference_steps",type=str,default=30,required=False,)
    parser.add_argument("--img_p_scale",type=float,default=0.70,required=False,)
    parser.add_argument("--ctrl_scale",type=float,default=0.8,required=False,)
    parser.add_argument("--txt_p_scale",type=float,default=7.5,required=False,)
    parser.add_argument("--seed",type=int,default=42,required=False,)
    parser.add_argument("--height",type=int,default=512,required=False,)
    parser.add_argument("--width",type=int,default=512,required=False,)
    parser.add_argument(
        "--adapter_path_if_ignore",
        default=["/workspace/Amiadapter-main/comic/models/checkpoint-90000.safetensors",True]
    )
    # parser.add_argument("--pos_prompt", type=str, default="a photo of a girl on the beach",)
    # parser.add_argument("--pos_prompt", type=str, default="a photo of a girl on the beach, wedding outfit",)
    # parser.add_argument("--neg_prompt", type=str, default="ugly, monochrome, two people",)
    parser.add_argument("--img_prompt1",default="assets/fortest_fullbody/788.png",)
    parser.add_argument("--img_prompt2",default="assets/fortest_halfbody/11_.png",)
    parser.add_argument("--img_prompt3",default="assets/fortest_halfbody/479.png",)
    parser.add_argument("--control_condition",default="assets/multipose/pose203.png",)
    parser.add_argument("--inpaint_image",default="assets/others/yangmi2.png",)
    # parser.add_argument("--control_condition",default="-pose/14pose.png",)
    parser.add_argument("--img_prompt_mask",default={"1_body": []})
    parser.add_argument("--img_prompt_mask_scale",default={"0_global":1,"1_body":1})# first value for whole, second value for mask
    args = parser.parse_args()
    return args

def record_config(args):
    f=open(os.path.join(args.output_dir,"config.txt"),"w")
    for k, v in sorted(vars(args).items()):
        f.write(k+": "+str(v)+"\n")
    f.close()

def main():
    global args
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    record_config(args)
    with gr.Blocks() as demo:
        demo.title = "Ami-adapter"
        with gr.Row():
            with gr.Column():
                c_pos = gr.Textbox(label="Positive prompts", value="a photo of a girl on the beach")
                c_neg = gr.Textbox(label="Negative prompts", value="ugly, monochrome, two people")
                with gr.Accordion("Open for More!", open=False):
                    with gr.Row():
                        c_bmodelpath = gr.Dropdown(["stablediffusionapi/anything-v5", "Lykon/DreamShaper",
                                                    "stablediffusionapi/counterfeit-v30","gsdf/Counterfeit-V2.5",
                                                    "stablediffusionapi/toonyou"], label="Base Model", info="Please select base model", value="stablediffusionapi/anything-v5")
                        # cXXX-unaug-90000 is trained with 7000 un-augmented samples for 90000 steps, 4*lambda GPU, 4 samples/GPU/step, about 67 hours
                        # cXXX-aug-82000 is finetuned on cXXX-45000(un-augmented data), with augmented data, 4*lambda GPU, 4 samples/GPU/step, about 67/9*8.2 hours totally
                        # cXXX-detail-85000 project image features into 1025 tokens, rather than 257, 4*lambda GPU, 4 samples/GPU/step, about ... hours totally
                        c_fmodelpath = gr.Dropdown(["checkpoint-unaug-90000.safetensors", "checkpoint-aug-82000.safetensors",
                                                            "checkpoint-detail-85000.safetensors"], label="Finetuned Model", info="Please select finetuned model", value="checkpoint-unaug-90000.safetensors")
                        c_tokennumbers=gr.Dropdown([257,1025], label="Number of Tokens", info="Please select num of tokens", value=257)
                    with gr.Row():
                        c_stepnum = gr.Number(label="Steps", value=args.num_inference_steps, minimum=0, maximum=100)
                        c_seednum = gr.Number(label="Seed", value=42, minimum=0, maximum=999999999999999999)
                        c_width = gr.Number(label="Width", value=512, minimum=8, maximum=999999999999999999)
                        c_height = gr.Number(label="Height", value=512, minimum=8, maximum=999999999999999999)
                        c_strength = gr.Number(label="Strength", value=1.0, minimum=0.0, maximum=999999999999999999)
                    with gr.Row():
                        c_ctrlnet=gr.Dropdown(["lllyasviel/sd-controlnet-openpose",
                                                   "lllyasviel/control_v11f1p_sd15_depth"], label="Ctrl net", info="Please select num of tokens", value="lllyasviel/sd-controlnet-openpose")
                        c_conditionornot = gr.Radio(["None", "Normal", "Condition"], label="Apply ctrl cond?", info="Pose selection", value="None")
                        c_inpaintornot = gr.Checkbox(label="Apply inpainting?")
                    with gr.Row():
                        scale1 = gr.Slider(label="img_p_scale", value=args.img_p_scale, minimum=0.0, maximum=1.0)
                        scale2 = gr.Slider(label="ctrl_scale", value=args.ctrl_scale, minimum=0.0, maximum=1.0)
                        scale3 = gr.Slider(label="txt_p_scale", value=args.txt_p_scale, minimum=0.0, maximum=10.0)
                    with gr.Row():
                        with gr.Column():
                            poseinput = gr.Image(label="Ctrlnet condition", height=256, width=256,
                                                value=args.control_condition, type="filepath")
                            btn_2condi = gr.Button(value="condi gen")
                        inpaintedimage = gr.ImageMask(label="Inpaint condition", value=args.inpaint_image,
                                                      brush=Brush(colors=["#7FDCDCDC"], color_mode="fixed"))


                with gr.Row():
                    apply_ref1 = gr.Checkbox(label="Apply this ref?")
                with gr.Row():
                    imginput1 = gr.ImageMask(label="Reference img", value=args.img_prompt1,
                                             brush=Brush(colors=["#7FDCDCDC"], color_mode="fixed"))
                    latentmask1 = gr.ImageMask(label="latentmask", value=Image.fromarray(
                        np.zeros([args.height, args.width, 3], dtype=np.uint8)),
                                               brush=Brush(colors=["#FFFFFF"], color_mode="fixed"))
                with gr.Row():
                    apply_ref2 = gr.Checkbox(label="Apply this ref?")
                with gr.Row():
                    imginput2 = gr.ImageMask(label="Reference img", value=args.img_prompt2,
                                             brush=Brush(colors=["#7FDCDCDC"], color_mode="fixed"))
                    latentmask2 = gr.ImageMask(label="latentmask", value=Image.fromarray(
                        np.zeros([args.height, args.width, 3], dtype=np.uint8)),
                                               brush=Brush(colors=["#FFFFFF"], color_mode="fixed"))
                with gr.Row():
                    apply_ref3 = gr.Checkbox(label="Apply this ref?")
                with gr.Row():
                    imginput3 = gr.ImageMask(label="Reference img", value=args.img_prompt3,
                                             brush=Brush(colors=["#7FDCDCDC"], color_mode="fixed"))
                    latentmask3 = gr.ImageMask(label="latentmask", value=Image.fromarray(
                        np.zeros([args.height, args.width, 3], dtype=np.uint8)),
                                               brush=Brush(colors=["#FFFFFF"], color_mode="fixed"))
            with gr.Column():
                out1 = gr.Image(label="Generated image", type="pil")
                with gr.Row():
                    out2_1 = gr.Image(label="Original image", type="pil")
                    out2_2 = gr.Image(label="Original image", type="pil")
                    out2_3 = gr.Image(label="Original image", type="pil")
        inputs = [c_bmodelpath, c_fmodelpath,c_tokennumbers, c_stepnum, c_seednum, c_pos, c_neg,c_width,c_height,c_strength,
                  c_ctrlnet, c_conditionornot,
                  c_inpaintornot,scale1, scale2, scale3,
                  imginput1,latentmask1,apply_ref1,imginput2,latentmask2,apply_ref2,imginput3,latentmask3,apply_ref3, poseinput,inpaintedimage]
        outputs = [out1, out2_1,out2_2,out2_3]

        btn = gr.Button(value="Run")
        btn.click(test, inputs=inputs, outputs=outputs)
        btn_2condi.click(condi_gen, inputs=[c_ctrlnet,poseinput,c_conditionornot], outputs=[latentmask1,latentmask2,latentmask3])

    demo.queue(max_size=1)
    demo.launch(share=True)

if __name__ == '__main__':
    main()
    
