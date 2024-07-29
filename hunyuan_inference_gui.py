from ast import Interactive
import gradio as gr
import hunyuan_test_lora
import hunyuan_test_lycoris
import hunyuan_test_dreambooth

def process_selection(selection, prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path, ckpt_path,
                      model_version):
    result = None

    if "Dreambooth" == selection:
        result = inference_by_dreambooth(prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path,
                                         ckpt_path, model_version)

    if "LoRA" == selection:
        result = inference_by_lora(prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path, ckpt_path,
                                   model_version)

    if "LyCORIS" == selection:
        result = inference_by_lycoris(prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path, ckpt_path,
                                      model_version)

    return "output.png"


def inference_by_dreambooth(prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path, ckpt_path,
                            model_version):
    hunyuan_test_dreambooth.generate_image(prompt, neg_prompt, seed, height, weight, steps, cfg_scale, model_path, ckpt_path, model_version)


def inference_by_lora(prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path, ckpt_path, model_version):
    hunyuan_test_lora.generate_image(prompt, neg_prompt, seed, height, weight, steps, cfg_scale, model_path, ckpt_path, model_version)


def inference_by_lycoris(prompt, neg_prompt, height, weight, steps, cfg_scale, seed, model_path, ckpt_path,
                         model_version):
    hunyuan_test_lycoris.generate_image(prompt, neg_prompt, seed, height, weight, steps, cfg_scale, model_path, ckpt_path, model_version)


fintune_options = ["Dreambooth", "LoRA", "LyCORIS"]
version_options = ["1.1", "1.2"]

iface = gr.Interface(
    process_selection,
    inputs=[
        gr.Radio(choices=fintune_options, label="Please select a training method.", value="Dreambooth"),
        gr.Textbox(label="prompt", value="1girl, long hair, looking at viewer, best quality", Interactive=True),
        gr.Textbox(label="neg_prompt", value="worst quality, bad quality错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺"),
        gr.Number(label="height", value=768),
        gr.Number(label="weight", value=1280),
        gr.Number(label="steps", value=40),
        gr.Number(label="cfg_scale", value=5),
        gr.Number(label="seed", value=287816226),
        gr.Textbox(label="model_path", value="/lv0/kohya_ss_hydit/models/HunyuanDiT-V1.2/t2i"),
        gr.Textbox(label="ckpt_path", value="/data/ckpt_path/.."),
        gr.Radio(choices=version_options, label="HunYuanDiT version", value="1.2"),
    ],
    outputs=gr.Image(label="Outputs"),
    title="HunyuanDIT Inference Tool Adapted for Kohya"
)

iface.launch(server_name="0.0.0.0", server_port=7888)