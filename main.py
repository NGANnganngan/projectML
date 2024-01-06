from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from IPython.display import HTML
import os
from gtts import gTTS
import torch
import torch.nn as nn
import model
import video_loader
import video_transformation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# prepare model

num_classes = 2000
i3d = model.InceptionI3d(400, in_channels=3)
i3d.replace_logits(num_classes)
i3d.load_state_dict(torch.load('FINAL_nslt_2000.pt'))
#i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
i3d.cuda()
i3d = nn.DataParallel(i3d)
i3d.eval()


app = Flask(__name__)
app.config['UPLOAD_DIRECTORY'] ='upload/'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 #1MB
app.config['ALLOWED_EXTENSIONS'] = ['.mp4']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1].lower()
        file_name = os.path.splitext(file.filename)[0]

        if file:
            if extension not in app.config['ALLOWED_EXTENSIONS']:
                return 'Please upload mp4 file.'
            
            vid_path = os.path.join(
                app.config['UPLOAD_DIRECTORY'],
                secure_filename(file.filename)
            )
            # vid_path = secure_filename(file.filename)
            file.save(vid_path)

            # preprocess data
            v_t = video_loader.data_reader(vid_path)
            v_t = v_t.unsqueeze(0)

            per_frame_logits = i3d(v_t)
            predictions = torch.max(per_frame_logits, dim=2)[0]
            out_labels = torch.argmax(predictions[0]).item()

            file_path = 'wlasl_class_list.txt'
            with open(file_path, 'r') as file:
                lines = file.readlines()
                predict_text = ''.join(c for c in lines[int(out_labels)] if c.isalpha())

            language ="en"
            speech = gTTS(text=predict_text, lang=language, slow=False, tld="com.au")
            audio_path=os.path.join(
                app.config['UPLOAD_DIRECTORY'],
                file_name + "_audio.mp3"
            )
            speech.save(audio_path)

    except RequestEntityTooLarge:
        return 'File is larger than 1MB limit.'

    # return render_template('index.html', video = video)
    return render_template('index.html', vid_path = file_name, predict_text = predict_text)#, audio_path = audio_path)

if __name__ == '__main__':
    app.run(debug=True)