# 恐怖！特大lvzhu来袭！

## 下载

前往[https://github.com/shiguangqwq0326/pjsk-auto-lvzhu/releases](https://github.com/shiguangqwq0326/pjsk-auto-lvzhu/releases/tag/v1.0.0)进行下载喵
下载并解压压缩包后按下方运行喵

## 关于运行

双击 `启动.bat`，输入一张图片的完整路径并回车。也可以将图片拖到 BAT 文件上。程序在**原图片所在文件夹**生成 `原文件名_overlay.gif`，原图片保持不变；再次处理同名图片会覆盖同名成品 GIF。

支持 PNG、JPG、JPEG、WEBP 和 BMP。没有识别出手时仍会生成一张静态 GIF。

文件夹中包含检测、分数修正、GIF 合成和校验脚本，校准数据、动画素材，以及当前 Windows 机器可直接使用的 Python 环境。`run_one.py` 是单图入口；它在内存中处理检测结果，只将成品 GIF 放在输入图片旁。复制到另一台电脑时，Python 环境可能需要按 `requirements.txt` 重建。

## 首次在另一台 Windows 电脑使用

复制整个 `lvzhu。！` 文件夹，包括 `assets` 和 `calibration`。随文件夹复制的 `.venv312` 与原电脑的 Python 安装有关；在新电脑上先安装 **64 位 Python 3.12**，然后打开 PowerShell，进入这个文件夹，建立本机环境并安装依赖：

```powershell
cd '这里替换成你存放 lvzhu。！ 的完整路径'
py -3.12 --version
py -3.12 -m venv .venv_local
& '.\.venv_local\Scripts\python.exe' -m pip install -r '.\requirements.txt'
& '.\.venv_local\Scripts\python.exe' -c "import PIL, onnxruntime, imgutils, sklearn; print('依赖检查通过')"
```

安装依赖需要联网。若系统找不到 `py`，请确认 Python 3.12 已安装并启用了 Python Launcher；也可以将以上两处 `py -3.12` 改成指向 Python 3.12 的完整路径。依赖检查通过后，双击 `启动.bat`：它会优先使用新建的 `.venv_local`，当前电脑则继续使用随副本复制的 `.venv312`。

**首次在没有模型缓存的电脑上识别图片时，也必须保持联网。**程序会从 Hugging Face 下载约 43 MB 的动漫手 ONNX 模型，保存在该电脑的用户缓存中；以后检测会复用缓存并可离线运行。若首次下载失败，请检查网络连接后重新运行 `启动.bat`。删除用户缓存后，下次运行仍需重新下载模型。

命令行也可以运行：

```powershell
& '.\.venv_local\Scripts\python.exe' '.\run_one.py' 'D:\path\图片.webp'
```

如果没有创建 `.venv_local`，把命令中的环境路径换成 `.\.venv312\Scripts\python.exe`。

当前规则：单手 GIF 倍率 0.88；两手中心距离不超过 200 像素时配对，双手 GIF 以两手中心为准，倍率 0.9；检测阈值 0.35。`calibration/reviewed_labels.json` 包含用户确认的右上角误检，`calibration/confidence_model.json` 是由这些标签生成的分数修正模型。原始 ONNX 检测器权重没有改动。
