from flask import Flask, request, jsonify
from flask_cors import CORS 
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from moviepy.editor import VideoFileClip

import time
import uuid

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

step = 8
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        prompt = request.json['prompt']

        # prompt 확인
        print(prompt)

        output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # 랜덤한 고유 ID 생성
        print('랜덤한 고유 ID 생성')
        unique_id = str(uuid.uuid4())[:8]  
        filename = f"animation_{timestamp}_{unique_id}"
        export_to_gif(output.frames[0], f"static/{filename}.gif")

        # gif > mp4 변환
        print('mp4 변환')
        gif_path = f"static/{filename}.gif"
        output_path = f"static/{filename}.mp4"
        clip = VideoFileClip(gif_path) 
        clip.write_videofile(output_path)

        # 클라이언트에게 생성된 비디오 전송
        print('전송')
        return jsonify({'url': f'http://localhost:5000/static/{filename}.mp4'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)