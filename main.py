import os
import glob
import sys
import cv2
import streamlit as st
import tempfile
import base64
from torchvision import transforms as pth_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import ml_collections
import vision_transformer as vits


def load_model():
    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(DEVICE)
    url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    if url is not None:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )
        model.load_state_dict(state_dict, strict=True)
    return model


FOURCC = {
    "mp4": cv2.VideoWriter_fourcc(*"MP4V"),
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
}
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VideoGenerator:
    def __init__(self, args):
        self.args = args
        self.model = self.args.model

    def run(self):
        if self.args.input_path is None:
            print(f"Provided input path {self.args.input_path} is non valid.")
            sys.exit(1)
        else:
            if self.args.video_only:
                self._generate_video_from_images(
                    self.args.input_path, self.args.output_path
                )
            else:
                # If input path exists
                if os.path.exists(self.args.input_path):
                    # If input is a video file
                    if os.path.isfile(self.args.input_path):
                        frames_folder = os.path.join(self.args.output_path, "frames")
                        attention_folder = os.path.join(
                            self.args.output_path, "attention"
                        )

                        os.makedirs(frames_folder, exist_ok=True)
                        os.makedirs(attention_folder, exist_ok=True)

                        self._extract_frames_from_video(
                            self.args.input_path, frames_folder
                        )

                        self._inference(
                            frames_folder,
                            attention_folder,
                        )

                        self._generate_video_from_images(
                            attention_folder, self.args.output_path
                        )

                    # If input is a folder of already extracted frames
                    if os.path.isdir(self.args.input_path):
                        attention_folder = os.path.join(
                            self.args.output_path, "attention"
                        )

                        os.makedirs(attention_folder, exist_ok=True)

                        self._inference(self.args.input_path, attention_folder)

                        self._generate_video_from_images(
                            attention_folder, self.args.output_path
                        )

                # If input path doesn't exists
                else:
                    print(f"Provided input path {self.args.input_path} doesn't exists.")
                    sys.exit(1)

    def _extract_frames_from_video(self, inp: str, out: str):
        vidcap = cv2.VideoCapture(inp)
        self.args.fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {inp} ({self.args.fps} fps)")
        print(f"Extracting frames to {out}")

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                os.path.join(out, f"frame-{count:04}.jpg"),
                image,
            )
            success, image = vidcap.read()
            count += 1

    def _generate_video_from_images(self, inp: str, out: str):
        img_array = []
        attention_images_list = sorted(glob.glob(os.path.join(inp, "attn-*.jpg")))

        # Get size of the first image
        with open(attention_images_list[0], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            size = (img.width, img.height)
            img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        print(f"Generating video {size} to {out}")

        for filename in tqdm(attention_images_list[1:]):
            with open(filename, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
                img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        out = cv2.VideoWriter(
            os.path.join(out, "video." + self.args.video_format),
            FOURCC[self.args.video_format],
            self.args.fps,
            size,
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print("Done")


    def _inference(self, inp: str, out: str):
        for img_path in tqdm(sorted(glob.glob(os.path.join(inp, "*.jpg")))):
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

        if self.args.resize is not None:
            transform = pth_transforms.Compose(
                [
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(self.args.resize),
                    pth_transforms.Normalize(
                        (123.675, 116.28, 103.53), (0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            transform = pth_transforms.Compose(
                [
                    pth_transforms.ToTensor(),
                    pth_transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )

        img = transform(img)

        # make the image divisible by the patch size
        w, h = (
            img.shape[1] - img.shape[1] % self.args.patch_size,
            img.shape[2] - img.shape[2] % self.args.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // self.args.patch_size
        h_featmap = img.shape[-1] // self.args.patch_size

        attentions = self.model.get_last_selfattention(img.to(DEVICE))

        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - self.args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            nn.functional.interpolate(
                th_attn.unsqueeze(0),
                scale_factor=self.args.patch_size,
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        )

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0),
                scale_factor=self.args.patch_size,
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        )

        # save attentions heatmaps
        fname = os.path.join(out, "attn-" + os.path.basename(img_path))
        plt.imsave(
            fname=fname,
            arr=sum(
                attentions[i] * 1 / attentions.shape[0]
                for i in range(attentions.shape[0])
            ),
            cmap="inferno",
            format="jpg",
        )


def st_ui():
    args = ml_collections.ConfigDict()
    args.model = load_model()
    args.patch_size = 16
    args.output_path = "./"
    args.resize = 512
    args.threshold = 0.6
    args.video_only = False
    args.fps = 30.0
    args.video_format = "mp4"
    uploaded_file = st.sidebar.file_uploader("Load your own Video", type=["mp4", "mpeg"])
    st.title("Extracting Attention Heatmaps :mag:")
    st.info("Based on Facebook Research's DINO, this is a modified version of "
            "the Self-Supervised Learning algorithm. The GIF below depicts it's working:  ")
    file_ = open("Corgi.gif", "rb")
    contents = file_.read()
    d1 = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{d1}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    st.markdown('#')
    st.subheader("Try it on your Own :point_down:")
    tff = tempfile.NamedTemporaryFile(delete=False)
    if uploaded_file is not None:
        tff.write(uploaded_file.read())
        st.sidebar.text("Using uploaded Sample")
        st.sidebar.video(str(tff.name))
        v = tff.name
    else:
        st.sidebar.text("Using default Sample")
        video_file = open('corgi.mp4', 'rb')
        video_bytes = video_file.read()  # reading the file
        st.sidebar.video(video_bytes)
        v = 'corgi.mp4'
    args.input_path = v
    b = st.button('Generate')
    if b:
        vg = VideoGenerator(args)
        with st.spinner('Usually takes 3-5 minutes...'):
            vg.run()
        st.success('Download available!')
        with open("video.mp4", "rb") as file:
            btn = st.download_button(
                label="Download Video",
                data=file,
                file_name="Attention Heatmap.mp4",
                mime="video/mp4"
            )


if __name__ == '__main__':
    st_ui()
