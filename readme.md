# Setup environment

Run the following commands to setup a python virtual env.

```
python -m venv .venv
pip install virtualenv
.venv\Scripts\activate
[linux]source .venv/bin/activate
pip install -r requirements.txt
```

Run the following command to download the SAM model on Linux:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth