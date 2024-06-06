一. 安装和配置环境

1. 安装本项目
`git clone https://github.com/echeloni11/couple_avatar.git

2. 安装diffusers
  ```git clone https://github.com/echeloni11/diffusers.git
  cd diffusers
  pip install -e ".[torch]"
```

  Notice: 这里安装的diffusers库是从https://github.com/huggingface/diffusers.git fork出来然后我进行了修改的。
  但目前代码无法运行，因为我实验的时候使用pip安装了diffusers库，然后是在conda环境的代码文件里直接修改了diffusers的代码，加入了加噪和重建的代码。
  为了将代码能够传播，我需要改用git安装并fork出来上传我做出的修改。然而，在我改用git安装diffusers时，它自动将我原先用pip安装的代码文件删除了，导致我自己修改过的代码丢失了。
  目前仍然在恢复中，还没有修改好，因此现在代码暂时无法运行

3. 配置环境
   在couple_avatar项目目录下
   `conda env create -f environment.yml

二. 运行实验
需要将提交的avatars64x64.zip放进couple_avatar目录并解压缩
DDPM加噪重建结果
`python cf_experiment_ddpm.py
DDIM加噪重建结果
`python cf_experiment_ddim.py
