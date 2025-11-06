依次执行123

1. hfd.sh 下载image net1k数据集  命令如下
./hfd.sh ILSVRC/imagenet-1k --hf_token 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  --dataset --local-dir 数据集保存位置 --hf_username WM33333

2. paraquest_2_jpg.py  将下载的parquet数据集压缩包 转化为jpg 
需要修改的参数
PARQUET_DIR 改为 数据集保存位置/data(parquet所在文件夹)
OUTPUT_ROOT = 改为数据集图片保存位置

3. render.py  render白底黑字的图片
需要改的参数
FONT_PATH改为字体位置

修改main函数中的
image_json_path     paraquest_2_jpg.py 输出的json位置
label_json_path     压缩包中的label_map.json文件位置
output_root 换为render结果图保存位置

预估时间：
从数据集文件中拿数据集图片还有生成render图片10w  items一共22min
