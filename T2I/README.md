两阶段执行，先执行load_t2i_2M.py  再执行fixed_font_size_from_txt.py  作用: 数据集图片与prompt图片


1. 编辑load_t2i_2M.py  作用: 从text-to-image-2M 数据集中获取图片与prompt
需要修改的参数: 
IMAGE_SAVE_DIR  ## **数据集图片的保存位置，作为训练用的output image path**
TEXT_SAVE_DIR  ## **prompt以txt格式保存位置**  fixed_font_size_from_txt.py 要用

2. 编辑fixed_font_size_from_txt.py    作用: 根据txt文件中的prompt  render为白底黑字的图片
需要修改的参数: 
FONT_PATH  ## **字体位置**
OUTPUT_DIR  ## **render结果图的保存位置，作为训练用的input image path**
TEXT_INPUT_DIR  ## **1中TEXT_SAVE_DIR的位置   就是txt所在文件夹的位置**

**预估时间：**
**获取图片**

10w条≈1h

**render image**

10w条≈1h