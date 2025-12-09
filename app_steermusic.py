import gradio as gr
import subprocess
import os
import uuid
import glob
from scipy.io.wavfile import read as wav_read
import numpy as np

# Main function that will be called when user hits "Run"
def steermusic_edit(audio_path, source_prompt, target_prompt,
                    guidance_scale, weight_aug,step_aug):
    if audio_path is None:
        return "Please upload a source music file.", None

    # audio_path is a filepath string on the Space
    out_dir = os.path.join("outputs", uuid.uuid4().hex)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "SteerMusic_edit.py",
        "--audio_path", audio_path,
        "--prompt", target_prompt,
        "--prompt_ref", source_prompt,
        "--output_dir", out_dir,
        "--guidance_scale", str(guidance_scale),
        "--weight_aug", str(weight_aug),
        "--validation_step", str(step_aug),
    ]

    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        return f"Error while running SteerMusic: {e}", None

    # SteerMusic_edit.py will save edited audio(s) into out_dir.
    # We just pick the first .wav we find.
    wavs = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
    if not wavs:
        return "No output audio was produced.", None

    # Load the first wav as (sample_rate, numpy_array)
    sr, data = wav_read(wavs[0])

    return "Success!", (sr, data)
    # wavs = sorted(glob.glob(os.path.join(out_dir, "*.wav")))
    # if not wavs:
    #     return "No output audio was produced.", None

    # return "Success!", wavs[0]


with gr.Blocks() as demo:
    gr.Markdown("## SteerMusic: Text-guided Music Editing Demo")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Source music"
        )

        with gr.Column():
            source_prompt = gr.Textbox(
                lines=2,
                label="Source prompt (describes the original music)",
                placeholder="e.g., Energetic piano cover with a groovy, reverberant melody."
            )
            target_prompt = gr.Textbox(
                lines=2,
                label="Target prompt (desired edit)",
                placeholder="e.g., Energetic harp cover with a groovy, reverberant melody."
            )
            guidance_scale = gr.Slider(
                minimum=5,
                maximum=50,
                value=30,
                step=1,
                label="Guidance scale"
            )
            weight_aug = gr.Slider(
                minimum=0,
                maximum=5,
                value=3,
                step=1,
                label="Weight aug"
            )
            step_aug = gr.Slider(
                minimum=10,
                maximum=500,
                value=300,
                step=1,
                label="Optimization step aug"
            )


    run_btn = gr.Button("Run SteerMusic")

    status = gr.Textbox(label="Status", interactive=False)
    audio_output = gr.Audio(label="Edited music", type="numpy")

    run_btn.click(
        fn=steermusic_edit,
        inputs=[audio_input, source_prompt, target_prompt, guidance_scale, weight_aug,step_aug],
        outputs=[status, audio_output],
    )

if __name__ == "__main__":
    demo.launch()